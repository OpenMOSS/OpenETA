# What is libuipc?

[libuipc](https://spirimirror.github.io/libuipc-doc/) is a **incremential potential contact** (IPC) framework. IPC ensures accurate, robust, penetration-free frictional contact, but IPC is computationally very expensive. libuipc is GPU accelerated and offers amazing performance compared to older IPC frameworks (still not comparable though to the simulation speed of PhysX).

This framework, unlike [GIPC](https://github.com/KemengHuang/GPU_IPC), can
simulate rigid bodies, soft bodies, cloth and threads, as well as their couplings.

Libuipc is a also fully differentiable and can be used for backward optimizations (the API is not done yet though).


# TacEx and Libuipc
We have a basic integration of libuipc in Isaac Sim.
The libuipc simulation runs alongside Isaac Sim.
The results of the libuipc sim are rendered into Isaac Sim by using the [USDRT](https://docs.omniverse.nvidia.com/kit/docs/usdrt/latest/docs/usd_fabric_usdrt.html) API:
- each libuipc objects gets an USD prim
- the mesh points of the prim mesh are updated with the newest position values from libuipc

> [!NOTE]
> The current implementation can probably be optimized a lot. Currently we iterate in a for loop through all the libuipc objects. In scenes with a lot of geometry this is taxing the simulation performance, e.g. the Wrecking Ball example.
> There is also probably a way to use the GPU buffers in a smarter way.
> We are going to work on this later. Later, cause the bottelneck is the IPC simulation.
>
> <details>
  <summary>Timing Breakdown Wreacking Ball example</summary>

```bash
=====================================================================================
                         TIMING BREAKDOWN: MERGED TIMERS
=====================================================================================
*GlobalTimer                                                                                      |       Time Cost |  Total Count
            *Pipeline                                                                             |   230.636282 ms |            1
                     *Simulation                                                                  |   230.622035 ms |            1
                                *Newton Iteration                                                 |    196.29653 ms |            7
                                                 *Compute Contact                                 |    82.796774 ms |            7
                                                 *Line Search                                     |    56.927383 ms |            6
                                                             *Detect Trajectory Candidates        |    38.636675 ms |            6
                                                             *Filter Contact Candidates           |     1.284685 ms |            6
                                                             *Filter CCD TOI                      |     1.174261 ms |            6
                                                             *Line Search Iteration               |     0.010685 ms |            6
                                                 *Detect DCD Candidates                           |    31.024746 ms |            6
                                                 *Solve Global Linear System                      |    25.206864 ms |            7
                                                                            *Build Linear System  |     18.98648 ms |            7
                                                                            *Solve Linear System  |     6.079092 ms |            7
                                *Detect DCD Candidates                                            |    22.201751 ms |            1
                                *Update Velocity                                                  |     0.029683 ms |            1
                     *Rebuild Scene                                                               |      0.00475 ms |            1

====================================================================================
====================================================================================
Step number  506
[INFO]: Time taken for uipc sim step : 0.234776 seconds
[INFO]: Time taken for rendering : 0.052545 seconds

```

</details>
