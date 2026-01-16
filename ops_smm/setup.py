import os
import glob
import torch
from torch import cuda
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension
from setuptools import setup, find_packages

requirements = ["torch", "torchvision"]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")  # Assuming source code is in src directory

    # Get C++ and CUDA source files
    source_cpp = glob.glob(os.path.join(extensions_dir, "*.cpp"))  # Can be ignored if no C++ files
    source_cuda = glob.glob(os.path.join(extensions_dir, "*.cu"))  # Find CUDA source files

    # Combine all source files
    sources = source_cpp + source_cuda
    extension = CppExtension  # Default to C++ extension
    extra_compile_args = {"cxx": []}
    define_macros = []

    # Switch to CUDA extension if CUDA is available
    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        define_macros += [("WITH_CUDA", None)]

        # Auto-detect CUDA architectures
        # Get all architectures supported by current CUDA environment
        cuda_archs = cuda.get_gencode_flags().replace('compute=', 'arch=').split()
        if not cuda_archs:
            print(f"Warning: Failed to auto-detect CUDA architectures")
            # Fallback to common architectures if auto-detection fails
            cuda_archs = [
                # "-gencode", "arch=compute_50,code=sm_50",  # Backward compatibility
                # "-gencode", "arch=compute_60,code=sm_60",
                # "-gencode", "arch=compute_70,code=sm_70",
                # "-gencode", "arch=compute_75,code=sm_75",
                "-gencode", "arch=compute_80,code=sm_80",
                "-gencode", "arch=compute_86,code=sm_86",
                "-gencode", "arch=compute_89,code=sm_89", # Comment this if CUDA version is too low
            ]
            print("Warning: Using fallback CUDA architectures")
       
        # Set CUDA compilation arguments
        extra_compile_args["nvcc"] = cuda_archs + [
            "-O3",  # Optimization level
            "-Xcompiler", "-Wall",  # Pass warning options to compiler
            "-lineinfo",  # Debug information
            # "--ptxas-options=-v",  # Verbose output (for debugging)
            # "--use_fast_math",  # Fast math operations
        ]
        
        print(f"Building with CUDA architectures: {cuda_archs}")
    else:
        raise NotImplementedError('CUDA is not available or not found.')

    # Ensure full paths for all source files
    sources = [os.path.join(extensions_dir, s) for s in sources]

    # Include directories for header files
    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            name="smm_cuda",  # Module name
            sources=sources,  # Source files
            include_dirs=include_dirs,  # Header directories
            define_macros=define_macros,  # Macro definitions
            extra_compile_args=extra_compile_args,  # Compilation options
        )
    ]
    return ext_modules


setup(
    name="smm_cuda",
    version="1.0",
    author="WeiLong",
    description="Sparse Matrix Multiplication (CUDA)",
    packages=find_packages(),
    ext_modules=get_extensions(),  # Get extension modules
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    install_requires=requirements,
)