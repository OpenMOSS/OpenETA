from __future__ import annotations

import torch
import weakref
from typing import TYPE_CHECKING

import carb
import omni.kit.app
from isaacsim.core.api.simulation_context import SimulationContext

from isaaclab.ui.widgets import ManagerLiveVisualizer

from .image_plot import ImagePlot
from .line_plot import LiveLinePlot

if TYPE_CHECKING:
    import omni.ui


class DirectLiveVisualizer(ManagerLiveVisualizer):
    def __init__(self, debug_vis: bool, num_envs: int, parent_window: omni.ui.Window, visualizer_name: str):
        """Initialize ManagerLiveVisualizer.

        Args:
            manager: The manager with terms to be plotted. The manager must have a get_active_iterable_terms method.
            cfg: The configuration file used to select desired manager terms to be plotted.
        """

        self.visualizer_name = visualizer_name
        self.debug_vis = debug_vis
        self.num_envs = num_envs
        self._env_idx: int = 0
        self._viewer_env_idx = 0
        self._vis_frame: omni.ui.Frame
        self._vis_window: omni.ui.Window = parent_window

        # evaluate chosen terms if no terms provided use all available.
        self._term_visualizers = {}
        self.terms: dict[str, torch.tensor] = {}
        self.terms_names: dict[str, list[str]] = {}

    @property
    def get_vis_frame(self) -> omni.ui.Frame:
        """Getter for the UI Frame object tied to this visualizer."""
        return self._vis_frame

    @property
    def get_vis_window(self) -> omni.ui.Window:
        """Getter for the UI Window object tied to this visualizer."""
        return self._vis_window

    def set_debug_vis(self, debug_vis: bool):
        """Set the debug visualization external facing function.

        Args:
            debug_vis: Whether to enable or disable the debug visualization.
        """
        self._set_debug_vis_impl(debug_vis)

    # def create_window_elements(self):
    #     """Creates window elements for each term of the visualizer

    #     """
    #     for name in self.terms:
    #         with self._vis_window.ui_window_elements["debug_frame"]:
    #             with self._vis_window.ui_window_elements["debug_vstack"]:
    #                 self._vis_window._create_debug_vis_ui_element(name, self)

    def create_visualizer(self):
        with self._vis_window.ui_window_elements["debug_frame"]:
            with self._vis_window.ui_window_elements["debug_vstack"]:
                self._vis_window._create_debug_vis_ui_element(self.visualizer_name, self)

    #
    # Implementations
    #

    def _set_env_selection_impl(self, env_idx: int):
        """Update the index of the selected environment to display.

        Args:
            env_idx: The index of the selected environment.
        """
        if env_idx > 0 and env_idx < self.num_envs:
            self._env_idx = env_idx
        else:
            carb.log_warn(f"Environment index is out of range (0,{self.num_envs}")

    def _set_vis_frame_impl(self, frame: omni.ui.Frame):
        """Updates the assigned frame that can be used for visualizations.

        Args:
            frame: The debug visualization frame.
        """
        self._vis_frame = frame

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set the debug visualization implementation.

        Args:
            debug_vis: Whether to enable or disable debug visualization.
        """

        if not hasattr(self, "_vis_frame"):
            raise RuntimeError("No frame set for debug visualization.")

        # Clear internal visualizers
        self._term_visualizers = {}
        self._vis_frame.clear()

        if debug_vis:
            # if enabled create a subscriber for the post update event if it doesn't exist
            if not hasattr(self, "_debug_vis_handle") or self._debug_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
        else:
            # if disabled remove the subscriber if it exists
            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None

            self._vis_frame.visible = False
            return

        self._vis_frame.visible = True

        with self._vis_frame:
            with omni.ui.VStack():
                # Add a plot in a collapsible frame for each term available
                # self._env_idx
                for name, values in self.terms.items():
                    frame = omni.ui.CollapsableFrame(
                        name,
                        collapsed=False,
                        style={"border_color": 0xFF8A8777, "padding": 4},
                    )
                    with frame:
                        value = values[self._env_idx]

                        terms_names = self.terms_names[name] if name in self.terms_names else None
                        # create line plot for single or multivariable signals
                        len_term_shape = len(value.shape)
                        if len_term_shape == 0:
                            value = value.reshape(1)
                        if len_term_shape <= 1:
                            plot = LiveLinePlot(
                                y_data=[[elem] for elem in value.T.tolist()],
                                plot_height=150,
                                show_legend=True,
                                legends=terms_names,
                            )
                            self._term_visualizers[name] = plot
                        # create an image plot for 2d and greater data (i.e. mono and rgb images)
                        elif len_term_shape == 2 or len_term_shape == 3:
                            image = ImagePlot(
                                image=value.cpu().numpy(),
                                label=name,
                            )
                            self._term_visualizers[name] = image
                        else:
                            carb.log_warn(
                                f"DirectLiveVisualizer: Term ({name}) is not a supported data type for visualization."
                            )
                    frame.collapsed = True
        self._debug_vis = debug_vis

    def _debug_vis_callback(self, event):
        """Callback for the debug visualization event."""

        if not SimulationContext.instance().is_playing():
            # Visualizers have not been created yet.
            return

        # get updated data and update visualization
        for name, values in self.terms.items():
            # E.g. terms = actions: Actions values have the shape (num_envs, num_actions).
            # This means we have `num_actions` amount of plots in our 'actions' timeserie.
            # To plot this, we need to pass over a list of lists.
            # Specifically, `num_actions` amount of lists, where each inner list contains the datapoint for the corresponding timeserie
            value = values[self._env_idx]
            if len(value.shape) == 0:
                value = value.reshape(1)

            vis = self._term_visualizers[name]
            if isinstance(vis, LiveLinePlot):
                vis.add_datapoint(value.T.tolist())
            elif isinstance(vis, ImagePlot):
                vis.update_image(value.cpu().numpy())

        # # get updated data and update visualization
        # for name, values in self.terms.items():
        #     # E.g. terms = actions: Actions values have the shape (num_envs, num_actions).
        #     # This means we have `num_actions` amount of plots in our 'actions' timeserie.
        #     # To plot this, we need to pass over a list of lists.
        #     # Specifically, `num_actions` amount of lists, where each inner list contains the datapoint for the corresponding timeserie
        #     value = values[self._env_idx]
        #     if len(value.shape) == 0:
        #         value = value.reshape(1)

        #     vis = self._term_visualizers[name]
        #     if isinstance(vis, LiveLinePlot):
        #         vis.add_datapoint(value.T.tolist())
        #     elif isinstance(vis, ImagePlot):
        #         vis.update_image(value.cpu().numpy())
