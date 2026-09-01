# TacEx - Tactile Extension for Isaac Sim/ Isaac Lab

- this is the main extension of TacEx
- contains classes for simulating a GelSight sensor inside Isaac Sim

## Overview
- subdirectories contain approaches for simulation GelSight Sensors
- Taxim: For Tactile RGB Simulation
- FOTS: For Marker Simulation
    - could theoretically also simulate Tactile RGB, but we didn't implement this yet

- the specific configs for certain sensors (such as the GelSight Mini) are in the tacex_assets extension (`tacex_assets/sensors/GelSight`)


>[!Note]
>We probably need to restructure this extension once we have multiple different types
>of tactile sensors. Currently its only for GelSight Type sensors
