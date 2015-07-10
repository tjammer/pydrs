from distutils.core import setup
from Cython.Build import cythonize

ext_modules = cythonize(
    "pydrs.pyx", language="c++",
    )
ext_modules[0].extra_compile_args.extend(
    ['-DOS_LINUX', '-DHAVE_USB', '-DHAVE_LIBUSB10'])
ext_modules[0].include_dirs.extend(
    ['/home/jammer/Downloads/drs-5.0.3/include/'])
setup(ext_modules=ext_modules)
