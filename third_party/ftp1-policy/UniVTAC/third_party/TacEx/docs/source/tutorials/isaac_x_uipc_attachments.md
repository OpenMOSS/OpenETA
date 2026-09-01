You can attach UIPC soft bodies to Isaac Sim rigid bodies to move the soft bodies kinematically.
First, the **attachment points** need to be computed. These are points of the tet mesh that are "close enough" to a Isaac rigid body (to be more precise, its collider actually). For these points we compute the world positions _they should have_ according to the current pose of the rigid body.

A simulation step then looks as follows:
- the rigid body moves and gets a new pose
- according to this position we prescribe what position the attachment points should have
- and the uipc objects move accordingly

Let's take a look at an example.

# Isaac X UIPC Attachments

## The Code
