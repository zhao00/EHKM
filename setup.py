# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "stage2_core",
        ["stage2_core.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="stage2_core",
    ext_modules=cythonize(extensions, language_level="3"),
)
