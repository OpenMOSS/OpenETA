# OpenETA Embodied Manipulation Context

This context defines the language used to connect grasp prediction, object
placement prediction, and robot end-effector motion in OpenETA.

## Visual Pointing

**Pointing Prompt**:
A complete natural-language instruction that asks a visual grounding model to
identify one or more image locations. Its wording determines whether the model
should identify any instance, every instance, or an instance selected by a
relational description. The instruction is preserved as authored rather than
being reduced to an object label.
_Avoid_: Target label, automatically wrapped prompt, visual question

**Pointing Image Set**:
An ordered collection of one or more images interpreted together by a visual
grounding model. A Pointing Prompt may refer to members by their displayed
one-based names, such as Image 1, while structured results identify members by
their zero-based Pointing Image Index.
_Avoid_: Unordered image batch, reference image only, target image only

**Pointing Image Index**:
The zero-based position of an image within a Pointing Image Set to which a
predicted point belongs. It identifies the point's image coordinate space, not
the identity of an object across images.
_Avoid_: Object identity, camera index, one-based image label

## Placement Geometry

**Placement Object**:
The object being grasped and moved into a predicted placement configuration.
_Avoid_: Target object, child object, grasp object

**Placement Region**:
The local, depth-observable geometry of the supporting or receiving object that
constrains a placement prediction.
_Avoid_: Empty location, base object, placement point

**Object Mask**:
A pixel-aligned binary image from the shared observation snapshot that both
scopes a targeted grasp result and selects the visible geometry of the placement
object. Nonzero pixels select the object.
_Avoid_: Target mask, grasp mask

**Placement Region Mask**:
A pixel-aligned binary image from the same RGBD observation that selects valid
local geometry around the intended placement region. Nonzero pixels select the
region. At the tool boundary it may be passed as a complete SAM3 segmentation
artifact; AnyPlace consumes its `mask_ref` and `source_image` fields and ignores
other detection metadata.
_Avoid_: Base mask, hole mask, empty-space mask

**Shared Scene Frame**:
The OpenCV camera frame in which grasp poses, object point clouds, and
placement transforms are expressed together so they can be composed without
first estimating a separate object frame.
_Avoid_: Implicit frame, mixed frame

**Shared Observation Snapshot**:
The single aligned RGBD capture and its camera intrinsics from which the object
mask, placement region mask, selected grasp candidate, and placement geometry
all derive.
_Avoid_: Latest frame, equivalent view, same camera only

**Depth Truncation**:
The preprocessing upper bound on positive metric depth along the camera Z-axis
used when selecting pixels for point-cloud construction. Samples at or beyond
the bound are discarded rather than clamped to the bound.
_Avoid_: Network range, Euclidean radius, depth clamping

**Camera-Frame Up Direction (`up_direction_camera`)**:
The gravity-opposing world-up direction expressed as a direction vector in the
Shared Scene Frame. It carries orientation but no camera position or complete
camera-to-world transform.
_Avoid_: Image up, camera Y-axis, gravity-down vector, camera extrinsics

**Object Placement Transform**:
An active rigid transform that moves the placement object's points from their
current scene-frame positions to their predicted placed positions.
_Avoid_: Absolute placement pose, object-to-base pose, pose matrix

**Pick Grasp Pose**:
The Normalized Grasp Pose chosen for placement composition, expressed in the
shared scene frame.
_Avoid_: Object-to-hand pose, relative grasp pose

**Selected Grasp**:
The single normalized grasp candidate chosen for placement composition together
with the provenance that binds it to its predictor and shared observation snapshot.
_Avoid_: Grasp envelope, bare candidate, all grasp candidates

**Greedy Grasp Candidate Policy**:
The score-ranked candidates from one successful grasp inference together with
the single active candidate. The highest-ranked candidate is active first, and
only a structured candidate-specific rejection advances to the next rank.
_Avoid_: Free candidate selection, automatic robot execution, unordered grasp set

**Grasp Source**:
The predictor identity, grasp-selection mode, RGB, depth, Object Mask, camera
intrinsics, and gripper context that contextualize a set of grasp candidates from
one Shared Observation Snapshot. It is established only by a successful targeted
grasp inference that produces candidates; a failed inference or source without an
Object Mask does not identify a Placement Object.
_Avoid_: Grasp input, source files, latest observation

**Place Grasp Pose**:
The Normalized Grasp Pose corresponding to the Selected Grasp after the object
placement transform is applied, expressed in the same shared scene frame and
retaining its predictor and gripper identity.
_Avoid_: Raw placement pose, object placement transform

**Placement Grasp Composition**:
The rigid-transform relationship that applies an object placement transform to
a pick grasp pose to obtain the corresponding place grasp pose.
_Avoid_: Placement inference, robot motion planning

**GraspNet Grasp Frame**:
The canonical gripper frame used natively by AnyGrasp and as the normalization
target for other grasp predictors. Its origin is the predicted grasp center,
X-axis is the approach direction, and Y-axis is the gripper open-close direction.
_Avoid_: Robot end-effector frame, TCP frame

**Contact-GraspNet Grasp Frame**:
The model-native Panda gripper base frame used by Contact-GraspNet, whose
X-axis is the gripper open-close direction and Z-axis is the approach direction.
It is distinct from the GraspNet Grasp Frame even when both are expressed in
the same camera frame.
_Avoid_: GraspNet grasp frame, robot end-effector frame, TCP frame

**GraspGenX Grasp Frame**:
The model-native configured gripper-base frame used by GraspGenX, whose Z-axis
is the approach direction and X-axis is the gripper closing direction. A pose
in this frame is distinct from its Normalized Grasp Pose representation.
_Avoid_: GraspNet grasp frame, robot end-effector frame, TCP frame

**Normalized Grasp Pose**:
A grasp prediction expressed in the Shared Scene Frame using the GraspNet
Grasp Frame convention, regardless of the predictor's model-native frame.
_Avoid_: Model-native pose, robot end-effector pose, world-frame pose

**Gripper Name (`gripper_name`)**:
The required canonical identifier of a configured gripper geometry used to
condition a grasp prediction. Valid values are exactly the gripper capabilities
advertised by the active predictor; an arbitrary or unregistered label does not
identify a usable gripper.
_Avoid_: Robot name, free-form gripper label, gripper type

**Robot End-Effector Frame**:
The robot-specific frame whose pose is consumed by motion planning and control.
_Avoid_: GraspNet frame, grasp center

**Grasp-to-EEF Calibration**:
The robot-specific rigid relationship between the GraspNet grasp frame and the
robot end-effector frame, including both orientation and origin offset.
_Avoid_: Camera extrinsics, object-to-hand pose

## Visual Grounding

**Image Point Grounding**:
The localization of a text-referred target as zero or more two-dimensional
points in a source image. It provides sparse visual evidence only; it does not
establish a mask, bounding box, depth, three-dimensional pose, or robot action.
_Avoid_: Segmentation, object detection, grasp point, world point
