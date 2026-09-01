# Troubleshooting
Collection of bugs and possible fixes for installation issues.


## Unable to install tacex_uipc
Here are some things you can try:
- **Clean Rebuild**: Remove the build folder `./source/tacex_uipc/build` and try to pip install tacex_uipc again
> you need to do such a clean rebuild, for example, when installing it inside a docker container fails
- If the libuipc compilation fails during the pip installation of tacex_uipc:
  - try to compile libuipc separately first (go to `./source/tacex_uipc/libuipc` and follow the [instructions](https://spirimirror.github.io/libuipc-doc/build_install/linux/))
  - then try to install tacex_uipc again (go to the root dir of `tacex_uipc`, i.e. `./source/tacex_uipc`, and use `pip install -e . -v`)

<!-- ```bash
- If you get an error that looks like this
 -- [libuipc] Check python module [pybind11] with [/home/dh/miniforge3/envs/env_isaaclab/bin/python3.10]
  Traceback (most recent call last):
    File "<string>", line 1, in <module>
  ModuleNotFoundError: No module named 'pybind11'
  -- [libuipc] pybind11 not found, try installing pybind11...
  Collecting pybind11
    Using cached pybind11-2.13.6-py3-none-any.whl.metadata (9.5 kB)
  Using cached pybind11-2.13.6-py3-none-any.whl (243 kB)
  Installing collected packages: pybind11
  Successfully installed pybind11-2.13.6
  -- [libuipc] [pybind11] installed successfully with [/home/dh/miniforge3/envs/env_isaaclab/bin/python3.10].
  Traceback (most recent call last):
    File "<string>", line 1, in <module>
  ModuleNotFoundError: No module named 'pybind11'
  CMake Error at src/pybind/CMakeLists.txt:10 (file):
    file FILE([TO_CMAKE_PATH|TO_NATIVE_PATH] path result) must be called with
    exactly three arguments.
```

then look if you have the path like `ENV PATH="/opt/cmake/bin:${PATH}"` set. Removing it fixed the issue for me (I suppose this gets in the way of the conda installed cmake? This is actually needed in the docker setup). #hmm idk tbh if this was the issue -->

## Other Errors
- `RuntimeError: Detected that PyTorch and torch_scatter were compiled with different CUDA versions.`
- `OSError: .../env_isaaclab/lib/python3.10/site-packages/torch_scatter/_version_cuda.so: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSsb`

-\> find the correct `torch_scatter` version [here](https://data.pyg.org/whl/) and install via `pip install torch-scatter -f [the_url_to_the_version]`
