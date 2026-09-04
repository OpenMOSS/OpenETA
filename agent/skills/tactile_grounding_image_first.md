---
name: tactile_grounding_image_first
description: Read tactile images before derived guidance and arbitrate conflicts explicitly.
task_patterns:
  - tactile observation
  - tactile evidence grounding
allowed_tools:
  - observe_images
  - observe_structured_guidance
version: "0.1"
---

# Tactile Grounding: Image First

Use this read-only procedural skill to interpret paired GelSight observations. Inspect the native
left and right tactile images first, then request the derived structured summary. If the two sources
conflict, say so explicitly and do not discard a clear image-supported judgment merely because the
summary disagrees. If the images are genuinely ambiguous or unavailable, abstain.

This candidate guides evidence use only. It is not a learned policy, ICL demonstration, memory,
action skill, or task-success predictor.
