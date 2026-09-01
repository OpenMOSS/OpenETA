"""MLP-based FOTS optical simulator for GF225 sensor.

This module implements optical simulation using an MLP neural network
that maps surface normals to RGB colors, replacing Taxim's polynomial
lookup table approach.
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchF
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from ..gelsight_simulator import GelSightSimulator

if TYPE_CHECKING:
    from .mlp_fots_sim_cfg import MLPFOTSSimulatorCfg
    from ...gelsight_sensor import GelSightSensor


class MLP(nn.Module):
    """Multi-layer perceptron for normal-to-color mapping.
    
    Takes input of shape (N, 5) where each row is [norm_x, norm_y, nx, ny, nz]
    and outputs (N, 3) RGB difference values.
    """

    dropout_p = 0.05

    def __init__(self, input_size=5, output_size=3, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        x = torchF.relu(self.fc1(x))
        x = self.drop(x)
        x = torchF.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.fc4(x)
        return x


class MLPFOTSSimulator(GelSightSimulator):
    """MLP-based FOTS optical simulator for GelSight sensors.
    
    This simulator converts depth maps to tactile RGB images using:
    1. Multi-scale Gaussian smoothing of the height map
    2. Surface normal computation from the smoothed height map
    3. MLP inference to map (x, y, nx, ny, nz) -> RGB difference
    4. Add background image to get final tactile RGB
    """

    cfg: MLPFOTSSimulatorCfg

    def __init__(self, sensor: GelSightSensor, cfg: MLPFOTSSimulatorCfg):
        self.sensor = sensor
        super().__init__(sensor=sensor, cfg=cfg)

    def _initialize_impl(self):
        """Initialize optical simulator assets and models."""
        if self.cfg.device is None:
            self._device = self.sensor.device
        else:
            self._device = self.cfg.device
        
        self._num_envs = self.sensor._num_envs
        
        # Initialize indentation depth buffer
        self._indentation_depth = torch.zeros(
            (self._num_envs,), device=self._device
        )

        # Load calibration assets from calib folder
        calib_path = Path(self.cfg.calib_folder_path)
        
        # Load background RGB image
        bg_img_path = calib_path / self.cfg.bg_img_filename
        if bg_img_path.exists():
            self.bg_img = np.load(str(bg_img_path))
        else:
            # Fallback: create gray background
            h, w = self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]
            self.bg_img = np.ones((h, w, 3), dtype=np.float32) * 128.0
            print(f"[MLPFOTSSimulator] Warning: bg_img not found at {bg_img_path}, using gray background")
        
        # Background depth will be captured from first frame
        self.bg_depth = None
        self._bg_depth_initialized = False

        # Load MLP model
        self.mlp = MLP().to(self._device)
        weights_path = calib_path / self.cfg.mlp_weights_filename
        if weights_path.exists():
            self.mlp.load_state_dict(
                torch.load(str(weights_path), map_location=self._device, weights_only=True)
            )
            self.mlp.eval()
        else:
            raise FileNotFoundError(
                f"MLP weights not found at {weights_path}. "
                f"Please ensure calibration files are present."
            )

        # Store image resolution for convenience
        self.img_width = self.cfg.tactile_img_res[0]
        self.img_height = self.cfg.tactile_img_res[1]

        # Create output buffer: shape (num_envs, H, W, 3), values in [0, 1]
        self._tactile_rgb = torch.zeros(
            (self._num_envs, self.img_height, self.img_width, 3),
            device=self._device,
            dtype=torch.float32,
        )
        
        # Set background as initial output (normalized to [0, 1])
        bg_tensor = torch.tensor(
            self.bg_img / 255.0, device=self._device, dtype=torch.float32
        )
        for i in range(self._num_envs):
            self._tactile_rgb[i] = bg_tensor

    # -------------------------------------------------------------------------
    # Helper methods for optical simulation
    # -------------------------------------------------------------------------

    def _padding(self, img: np.ndarray) -> np.ndarray:
        """Pad image with symmetric boundary conditions."""
        if len(img.shape) == 2:
            return np.pad(img, ((1, 1), (1, 1)), mode="symmetric")
        elif len(img.shape) == 3:
            return np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="symmetric")
        return img

    def _generate_normals(self, height_map: np.ndarray) -> np.ndarray:
        """Generate surface normals from height map.
        
        Args:
            height_map: 2D array of height values (H, W)
            
        Returns:
            Surface normals as (H, W, 3) array with values in [0, 1]
        """
        h, w = height_map.shape
        
        # Compute finite differences
        top = height_map[0:h-2, 1:w-1]
        bot = height_map[2:h, 1:w-1]
        left = height_map[1:h-1, 0:w-2]
        right = height_map[1:h-1, 2:w]

        dzdx = (bot - top) / 2.0
        dzdy = (right - left) / 2.0

        # Build direction vectors (pointing up in z)
        direction = np.ones((h - 2, w - 2, 3), dtype=np.float32)
        direction[:, :, 0] = -dzdy
        direction[:, :, 1] = dzdx

        # Normalize to unit vectors
        magnitude = np.sqrt(
            direction[:, :, 0] ** 2 + 
            direction[:, :, 1] ** 2 + 
            direction[:, :, 2] ** 2
        )
        normal = direction / magnitude[:, :, np.newaxis]

        # Pad to original size and normalize to [0, 1]
        normal = self._padding(normal)
        normal = (normal + 1.0) * 0.5
        return normal

    def _smooth_height_map(self, height_map: np.ndarray) -> np.ndarray:
        """Smooth height map using multi-scale Gaussian blurring.
        
        Args:
            height_map: Raw depth/height map from camera
            
        Returns:
            Smoothed height map
        """
        if self.bg_depth is None:
            return height_map
            
        # Compute difference from background
        diff_depth = np.abs(height_map - self.bg_depth)
        diff_depth[np.abs(diff_depth) < 6e-5] = 0.0

        # Scale to appropriate range
        diff_depth *= 1000.0
        diff_depth /= (0.05107 * 2)

        # Create contact mask for preserving contact regions
        max_diff = np.max(diff_depth)
        if max_diff > 0:
            contact_mask = diff_depth > (max_diff * 0.4)
        else:
            contact_mask = np.zeros_like(diff_depth, dtype=bool)
            
        height_map = diff_depth
        zq_back = height_map.copy()

        # Multi-scale Gaussian blur while preserving contact regions
        kernel_sizes = [101, 51, 21, 11, 5]
        for k in kernel_sizes:
            height_map = cv2.GaussianBlur(
                height_map.astype(np.float32), (k, k), 0
            )
            height_map[contact_mask] = zq_back[contact_mask]

        # Final smoothing
        height_map = cv2.GaussianBlur(
            height_map.astype(np.float32), (5, 5), 0
        )
        return height_map

    def _preproc_mlp(self, normal: np.ndarray) -> torch.Tensor:
        """Preprocess normals for MLP input.
        
        Args:
            normal: Surface normal map (H, W, 3) with values in [0, 1]
            
        Returns:
            Torch tensor ready for MLP input (N, 5) where N = H * W
        """
        h, w = normal.shape[:2]

        # Create normalized coordinate grids
        rows, cols = np.indices((h, w))
        n_pixels = h * w
        
        xy_coords = np.stack([rows.ravel(), cols.ravel()], axis=1)
        nxyz = normal.reshape(n_pixels, 3)
        
        # Normalize coordinates to [0, 1]
        norm_x = xy_coords[:, 0] / float(h)
        norm_y = xy_coords[:, 1] / float(w)

        # Stack into (N, 5) array: [norm_x, norm_y, nx, ny, nz]
        input_data = np.column_stack((norm_x, norm_y, nxyz))
        return torch.tensor(input_data, dtype=torch.float32, device=self._device)

    # -------------------------------------------------------------------------
    # Main simulation methods
    # -------------------------------------------------------------------------

    def optical_simulation(self) -> torch.Tensor:
        """Simulate tactile RGB image from depth data.
        
        Returns:
            Torch tensor of shape (num_envs, H, W, 3) containing RGB images
            with values in [0, 1] range (following TaximSimulator convention).
        """
        # Get height map from sensor
        if "height_map" not in self.sensor._data.output:
            # Return background image if no height data
            print("[MLPFOTSSimulator] Warning: height_map not available from sensor.")
            return self._tactile_rgb
            
        height_map = self.sensor._data.output["height_map"]
        
        if height_map is None:
            # Return background image if no height data
            return self._tactile_rgb

        # Process each environment
        for env_idx in range(self._num_envs):
            # Extract height map for this environment
            depth_tensor = height_map[env_idx]
            if depth_tensor.dim() == 3:
                depth_tensor = depth_tensor.squeeze(-1)

            height_map = depth_tensor.cpu().numpy()
            
            # Initialize background depth from first frame
            if not self._bg_depth_initialized:
                self.bg_depth = height_map.copy()
                self._bg_depth_initialized = True

            # Processing pipeline: smooth -> normals -> MLP
            smoothed = self._smooth_height_map(height_map)
            normal = self._generate_normals(smoothed)
            mlp_input = self._preproc_mlp(normal)

            # MLP inference
            with torch.no_grad():
                mlp_output = self.mlp(mlp_input)

            # Post-process: convert to RGB
            sim_img_r = mlp_output.cpu().numpy()
            out_diff = (sim_img_r * 2 - 1) * 255  # Scale to [-255, 255]
            
            h, w = smoothed.shape
            sim_img = out_diff.reshape(h, w, 3).astype(np.float32)
            sim_img = sim_img + self.bg_img  # Add background
            sim_img = np.clip(sim_img, 0, 255)
            
            # Normalize to [0, 1] for TaximSimulator compatibility
            self._tactile_rgb[env_idx] = torch.tensor(
                sim_img / 255.0, device=self._device, dtype=torch.float32
            )

        return self._tactile_rgb

    def marker_motion_simulation(self):
        """Marker motion simulation is not implemented for MLP-FOTS.
        
        Use ManiSkillSimulator for marker motion if needed.
        """
        raise NotImplementedError(
            "MLPFOTSSimulator only provides optical_simulation. "
            "Use ManiSkillSimulator for marker_motion_simulation."
        )

    def compute_indentation_depth(self) -> torch.Tensor:
        """Compute indentation depth from height map.
        
        Returns:
            Tensor of shape (num_envs,) with indentation depths in mm.
        """
        # TODO: Implement proper indentation depth calculation
        return self._indentation_depth

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset the simulator state.
        
        Args:
            env_ids: Optional tensor of environment indices to reset.
                    If None, resets all environments.
        """
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        
        # Reset indentation depth
        self._indentation_depth[env_ids] = 0.0
        
        # Reset tactile RGB to background
        bg_tensor = torch.tensor(
            self.bg_img / 255.0, device=self._device, dtype=torch.float32
        )
        for idx in env_ids:
            self._tactile_rgb[idx] = bg_tensor

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Setup debug visualization for optical simulation."""
        if debug_vis and not hasattr(self, "_debug_windows"):
            self._debug_windows = {}
            self._debug_img_providers = {}

    def _debug_vis_callback(self, event):
        """Debug visualization callback for showing tactile RGB output."""
        # Import here to avoid circular imports
        try:
            import omni.ui
        except ImportError:
            return
            
        if self.sensor._prim_view is None:
            return

        for i, prim in enumerate(self.sensor._prim_view.prims):
            if "tactile_rgb" not in self.sensor.cfg.data_types:
                continue

            attr = prim.GetAttribute("debug_tactile_rgb")
            if not attr.IsValid():
                continue

            show_img = attr.Get()
            if show_img:
                if str(i) not in self._debug_windows:
                    self._debug_windows[str(i)] = omni.ui.Window(
                        f"{self.sensor._prim_view.prim_paths[i]}/mlp_fots_sim",
                        width=self.cfg.tactile_img_res[0],
                        height=self.cfg.tactile_img_res[1],
                    )
                    self._debug_img_providers[str(i)] = omni.ui.ByteImageProvider()

                # Get image data (convert from [0,1] to [0,255])
                img_data = self._tactile_rgb[i] * 255.0
                img_np = img_data.cpu().numpy().astype(np.uint8)

                # Convert to RGBA
                if img_np.shape[-1] == 3:
                    alpha = np.full(
                        (img_np.shape[0], img_np.shape[1], 1), 255, dtype=np.uint8
                    )
                    img_np = np.concatenate([img_np, alpha], axis=-1)

                # Update provider
                self._debug_img_providers[str(i)].set_bytes_data(
                    img_np.flatten().tolist(),
                    [img_np.shape[1], img_np.shape[0]]
                )

                with self._debug_windows[str(i)].frame:
                    omni.ui.ImageWithProvider(self._debug_img_providers[str(i)])

            elif str(i) in self._debug_windows:
                self._debug_windows[str(i)].visible = False
