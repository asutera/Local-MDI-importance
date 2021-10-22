"""

Author: Antonio Sutera (sutera.antonio@gmail.com)
License: BSD 3 clause

"""

# To compile (in the directory):
# > python setuplocalmdi.py build_ext --inplace

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("LocalMDI_cy.pyx")
)
