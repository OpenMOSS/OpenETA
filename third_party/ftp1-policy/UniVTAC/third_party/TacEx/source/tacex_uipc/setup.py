"""Installation script for the 'tacex_uipc' python package.

Invoke with, for example:
# in .../source/tacex_uipc folder
pip install -e .

or with user arguments:
pip install -C--build-option=build_ext -C--build-option=--DCMAKE-CUDA-ARCHITECTURES=89 -C--build-option=--DUIPC-BUILD-PYBIND=1 .

-> custom build arguments don't work for editable install, due to build_temp method (see https://github.com/pypa/setuptools/issues/2491)
Workaround: use ENV variables (see https://github.com/pypa/setuptools/discussions/3969)
"""

import os
import platform
import subprocess
import sys
import toml

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # NOTE: Add dependencies
    # "cmake>=3.26",
    # "pybind11"
    "wildmeshing>=0.4.1"
]


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    # hyphens are automatically converted to underscores for attribute values
    # -> underscore directly is not allowed for user options string
    user_options = (
        build_ext.user_options
        + [("DCMAKE-CUDA-ARCHITECTURES=", None, "Specify CUDA architecture,e.g. 89.")]
        + [("DUIPC-BUILD-PYBIND=", "1", "Whether to build libuipc python bindings or not.")]
    )

    def initialize_options(self):
        super().initialize_options()
        self.DCMAKE_CUDA_ARCHITECTURES = None
        self.DUIPC_BUILD_PYBIND = "1"

    def finalize_options(self):
        # assert self.DCMAKE_CUDA_ARCHITECTURES in (None)
        super().finalize_options()

    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # use ENV variable as workaround
        self.DCMAKE_CUDA_ARCHITECTURES = os.environ.get("CMAKE_CUDA_ARCHITECTURES")
        print("cuda_architectures", self.DCMAKE_CUDA_ARCHITECTURES)

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=1",
            "-DCMAKE_COLOR_DIAGNOSTICS=1",
            "-DUIPC_BUILD_PYBIND=" + self.DUIPC_BUILD_PYBIND,  # per default = 1, i.e. true
            "-DUIPC_DEV_MODE=1",
            "-DUIPC_BUILD_GUI=0",
        ]
        # Pass vcpkg toolchain explicitly so pip build (possibly isolated) and stale CMakeCache don't use a wrong path
        toolchain = os.environ.get("CMAKE_TOOLCHAIN_FILE")
        if toolchain and os.path.isfile(toolchain):
            cmake_args += ["-DCMAKE_TOOLCHAIN_FILE=" + os.path.abspath(toolchain)]
        if self.DCMAKE_CUDA_ARCHITECTURES is not None:  # None means "use native cuda architecture"
            cmake_args += ["-DCMAKE_CUDA_ARCHITECTURES=" + self.DCMAKE_CUDA_ARCHITECTURES]

        cfg = "Debug" if self.debug else "Release"
        build_args = []  # ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["-j4"]  # use -j8 for faster building

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get("CXXFLAGS", ""), self.distribution.get_version())
        self.build_dir = "build/"  # where the compiled files are placed
        if not os.path.exists(self.build_dir):
            os.makedirs(self.build_dir)
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_dir, env=env)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_dir)


# Installation operation
setup(
    name="tacex_uipc",
    packages=["tacex_uipc"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.5.0",
    ],
    ext_modules=[CMakeExtension("uipc", "libuipc")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
