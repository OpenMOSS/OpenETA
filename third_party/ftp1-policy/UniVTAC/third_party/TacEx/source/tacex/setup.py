"""Installation script for the 'tacex' python package."""

import os
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # NOTE: Add dependencies
    "debugpy",  # for debugging scripts that are run from the terminal, e.g. RL training scripts
    # "torch==2.5.1",
    # "torchvision==0.20.1",
    # (  # needed for gpu taxim
    #     "torch_scatter @"
    #     " https://data.pyg.org/whl/torch-2.5.0%2Bcu118/torch_scatter-2.1.2%2Bpt25cu118-cp310-cp310-linux_x86_64.whl"
    # ),
    (  # needed for gpu taxim
        "torch_scatter @"
        "https://data.pyg.org/whl/torch-2.8.0%2Bcu128/torch_scatter-2.1.2%2Bpt28cu128-cp310-cp310-linux_x86_64.whl"
    ),
    # (  # needed for gpu taxim -> for Isaac 5.0
    #     "torch_scatter @"
    #     "https://data.pyg.org/whl/torch-2.8.0%2Bcu128/torch_scatter-2.1.2%2Bpt28cu128-cp311-cp311-linux_x86_64.whl"
    # ),
    "psutil",
    "nvidia-ml-py",
    "pre-commit",
]

# PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu121"]

# Installation operation
setup(
    name="tacex",
    packages=["tacex"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    # dependency_links=PYTORCH_INDEX_URL,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 2023.1.1",
        "Isaac Sim :: 4.5.0",
    ],
    zip_safe=False,
)
