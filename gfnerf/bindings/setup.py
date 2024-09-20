import os
import re
import subprocess

import torch
from pkg_resources import parse_version
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

bindings_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(bindings_dir))
print('root dir:',root_dir)
base_source_files = [	

	# tcnn
	"%s/External/tiny-cuda-nn/src/common_device.cu" % root_dir,
	"%s/External/tiny-cuda-nn/src/common.cu" % root_dir,
	"%s/External/tiny-cuda-nn/src/cpp_api.cu" % root_dir,
	"%s/External/tiny-cuda-nn/src/encoding.cu" % root_dir,
	"%s/External/tiny-cuda-nn/src/fully_fused_mlp.cu" % root_dir,
	"%s/External/tiny-cuda-nn/src/loss.cu" % root_dir,
	"%s/External/tiny-cuda-nn/src/network.cu" % root_dir,
	"%s/External/tiny-cuda-nn/src/object.cu" % root_dir,
	"%s/External/tiny-cuda-nn/src/optimizer.cu" % root_dir,
	"%s/External/tiny-cuda-nn/src/reduce_sum.cu" % root_dir,
	"%s/External/tiny-cuda-nn/src/cutlass_mlp.cu" % root_dir,

	# fmt
	"%s/External/tiny-cuda-nn/dependencies/fmt/src/format.cc" % root_dir,
	"%s/External/tiny-cuda-nn/dependencies/fmt/src/os.cc" % root_dir,
	
	# hash3danchored
    "hashanchored/bindings.cpp",
	"field/Hash3DAnchored.cpp",
	"field/Hash3DAnchored_cuda.cu",
	"field/TCNNWP.cpp",
	"field/FieldFactory.cpp",
	# Utils
	"Utils/Pipe.cpp",
	"Utils/StopWatch.cpp",
	# perssampler
	"PtsSampler/PersSampler.cpp",
	"PtsSampler/PersSampler_cuda.cu"


]



include_dirs=[
    "%s/External/tiny-cuda-nn/include" % root_dir,
    "%s/External/tiny-cuda-nn/dependencies" % root_dir,
    "%s/External/tiny-cuda-nn/dependencies/cutlass/include" % root_dir,
    "%s/External/tiny-cuda-nn/dependencies/cutlass/tools/util/include" % root_dir,
    "%s/External/tiny-cuda-nn/dependencies/fmt/include" % root_dir,
    "%s/External/eigen-3.4.0" % root_dir,
	"field",
    "PtsSampler",
    "Utils"
]

def min_supported_compute_capability(cuda_version):
	if cuda_version >= parse_version("12.0"):
		return 50
	else:
		return 20

def max_supported_compute_capability(cuda_version):
	if cuda_version < parse_version("11.0"):
		return 75
	elif cuda_version < parse_version("11.1"):
		return 80
	elif cuda_version < parse_version("11.8"):
		return 86
	else:
		return 90
	
if torch.cuda.is_available():
	major, minor = torch.cuda.get_device_capability()
	compute_capabilities = [major * 10 + minor]
	print(f"Obtained compute capability {compute_capabilities[0]} from PyTorch")
	
# Get CUDA version and make sure the targeted compute capability is compatible
if os.system("nvcc --version") == 0:
	nvcc_out = subprocess.check_output(["nvcc", "--version"]).decode()
	cuda_version = re.search(r"release (\S+),", nvcc_out)

	if cuda_version:
		cuda_version = parse_version(cuda_version.group(1))
		print(f"Detected CUDA version {cuda_version}")
		supported_compute_capabilities = [
			cc for cc in compute_capabilities if cc >= min_supported_compute_capability(cuda_version) and cc <= max_supported_compute_capability(cuda_version)
		]

		if not supported_compute_capabilities:
			supported_compute_capabilities = [max_supported_compute_capability(cuda_version)]

		if supported_compute_capabilities != compute_capabilities:
			print(f"WARNING: Compute capabilities {compute_capabilities} are not all supported by the installed CUDA version {cuda_version}. Targeting {supported_compute_capabilities} instead.")
			compute_capabilities = supported_compute_capabilities
min_compute_capability = min(compute_capabilities)

base_nvcc_flags = [
	"-std=c++17",
	"--extended-lambda",
	"--expt-relaxed-constexpr",
    # since TCNN requires half-precision operation.
	"-U__CUDA_NO_HALF_OPERATORS__",
	"-U__CUDA_NO_HALF_CONVERSIONS__",
	"-U__CUDA_NO_HALF2_OPERATORS__",
    
    ]
base_definitions = ["-DHALF_PRECISION",]

if os.name == "posix":
	base_cflags = ["-std=c++17"]
	base_nvcc_flags += [
		"-Xcompiler=-Wno-float-conversion",
		"-Xcompiler=-fno-strict-aliasing",
	]
elif os.name == "nt":
	base_cflags = ["/std:c++14"]

# Some containers set this to contain old architectures that won't compile. We only need the one installed in the machine.
os.environ["TORCH_CUDA_ARCH_LIST"] = ""

def make_extension(compute_capability):
    nvcc_flags = base_nvcc_flags + [f"-gencode=arch=compute_{compute_capability},code={code}_{compute_capability}" for code in ["compute", "sm"]]
    definitions = base_definitions + [f"-DTCNN_MIN_GPU_ARCH={compute_capability}"]
    nvcc_flags = nvcc_flags + definitions
    cflags = base_cflags + definitions

    ext = CUDAExtension(
        # name=f"hash3danchored_bindings._{compute_capability}_C",
		name = 'f2nerf-bindings',
        sources=base_source_files,
        include_dirs=include_dirs,
	
        extra_compile_args={"cxx": cflags, "nvcc": nvcc_flags},
        libraries=["cuda"],
    )
    return ext

ext_modules = [make_extension(comp) for comp in compute_capabilities]


setup(
	name = 'f2nerf-bindings',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)}


)
