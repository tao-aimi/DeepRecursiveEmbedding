import os

import numpy


def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration("DRE", parent_package, top_path)

    libraries = []
    if os.name == "posix":
        libraries.append("m")

    config.add_extension(
        "_utils_tsne",
        sources=["_utils_tsne.pyx"],
        include_dirs=[numpy.get_include()],
        libraries=libraries,
        extra_compile_args=["-O3"],
    )

    config.add_subpackage("umap_xinrui")

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration().todict())