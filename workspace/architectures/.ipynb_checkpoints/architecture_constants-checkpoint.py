from enum import Enum

# The 3 architectures we are working with
class Architecture(Enum):
    Base = 'Base'
    Data_Parallel = 'Data_Parallel'
    Tensor_Parallel = 'Tensor_Parallel'    

class GPUMemoryScale(Enum):
    MEMORY_4MB = 'designs/system/small_arch.yaml'
    MEMORY_16MB = 'designs/system/medium_arch.yaml'
    MEMORY_64MB = 'designs/system/large_arch.yaml'
    MEMORY_1024MB = 'designs/system/xlarge_arch.yaml'

    @property
    def size_label(self) -> str:
        return {
            "MEMORY_4MB": "4MB",
            "MEMORY_16MB": "16MB",
            "MEMORY_64MB": "64MB",
            "MEMORY_1024MB": "1024MB",
        }[self.name]

class RackSize(Enum):
    RACK_2 = 2
    RACK_4 = 4
    RACK_8 = 8

ARCH_CONFIG = {'pe_meshX': 4, 'pe_meshY': 4}

# Base configs keep everything in DRAM and have global_buffer_factor set to 1
base_config = [
    dict(
        DRAM_factor_N=64,
        DRAM_factor_M=64,
        DRAM_factor_C=3,
        global_buffer_factor_N=1,
        global_buffer_factor_M=1,
        global_buffer_factor_C=1,
        PE_spatial_factor_N=1,
        PE_spatial_factor_M=1,
        PE_spatial_factor_C=1,
        scratchpad_factor_N=1,
         **ARCH_CONFIG,
    ),
    dict(
        DRAM_factor_N=64,
        DRAM_factor_M=64,
        DRAM_factor_C=64,
        global_buffer_factor_N=1,
        global_buffer_factor_M=1,
        global_buffer_factor_C=1,
        PE_spatial_factor_N=1,
        PE_spatial_factor_M=1,
        PE_spatial_factor_C=1,
        scratchpad_factor_N=1,
         **ARCH_CONFIG,
    ),
    dict(
        DRAM_factor_N=64,
        DRAM_factor_M=64,
        DRAM_factor_C=64,
        global_buffer_factor_N=1,
        global_buffer_factor_M=1,
        global_buffer_factor_C=1,
        PE_spatial_factor_N=1,
        PE_spatial_factor_M=1,
        PE_spatial_factor_C=1,
        scratchpad_factor_N=1,
         **ARCH_CONFIG,
    ),
    dict(
        DRAM_factor_N=64,
        DRAM_factor_M=128,
        DRAM_factor_C=64,
        global_buffer_factor_N=1,
        global_buffer_factor_M=1,
        global_buffer_factor_C=1,
        PE_spatial_factor_N=1,
        PE_spatial_factor_M=1,
        PE_spatial_factor_C=1,
        scratchpad_factor_N=1,
         **ARCH_CONFIG,
    ),
    dict(
        DRAM_factor_N=64,
        DRAM_factor_M=128,
        DRAM_factor_C=128,
        global_buffer_factor_N=1,
        global_buffer_factor_M=1,
        global_buffer_factor_C=1,
        PE_spatial_factor_N=1,
        PE_spatial_factor_M=1,
        PE_spatial_factor_C=1,
        scratchpad_factor_N=1,
         **ARCH_CONFIG,
    ),
    dict(
        DRAM_factor_N=64,
        DRAM_factor_M=256,
        DRAM_factor_C=128,
        global_buffer_factor_N=1,
        global_buffer_factor_M=1,
        global_buffer_factor_C=1,
        PE_spatial_factor_N=1,
        PE_spatial_factor_M=1,
        PE_spatial_factor_C=1,
        scratchpad_factor_N=1,
         **ARCH_CONFIG,
    ),
    dict(
        DRAM_factor_N=64,
        DRAM_factor_M=256,
        DRAM_factor_C=256,
        global_buffer_factor_N=1,
        global_buffer_factor_M=1,
        global_buffer_factor_C=1,
        PE_spatial_factor_N=1,
        PE_spatial_factor_M=1,
        PE_spatial_factor_C=1,
        scratchpad_factor_N=1,
         **ARCH_CONFIG,
    ),
    dict(
        DRAM_factor_N=64,
        DRAM_factor_M=512,
        DRAM_factor_C=256,
        global_buffer_factor_N=1,
        global_buffer_factor_M=1,
        global_buffer_factor_C=1,
        PE_spatial_factor_N=1,
        PE_spatial_factor_M=1,
        PE_spatial_factor_C=1,
        scratchpad_factor_N=1,
         **ARCH_CONFIG,
    ),
    dict(
        DRAM_factor_N=64,
        DRAM_factor_M=512,
        DRAM_factor_C=512,
        global_buffer_factor_N=1,
        global_buffer_factor_M=1,
        global_buffer_factor_C=1,
        PE_spatial_factor_N=1,
        PE_spatial_factor_M=1,
        PE_spatial_factor_C=1,
        scratchpad_factor_N=1,
         **ARCH_CONFIG,
    ),
]

# Each index represents the corresponding layer
layer_shapes = [
    dict(
        C=3,
        M=64,
        R=7,
        S=7,
        P=112,
        Q=112,
        N=64,
    ),
    dict(
        C=64,
        M=64,
        R=3,
        S=3,
        P=56,
        Q=56,
        N=64,
    ),
    dict(
        C=64,
        M=64,
        R=3,
        S=3,
        P=56,
        Q=56,
        N=64,
    ),
    dict(
        C=64,
        M=128,
        R=3,
        S=3,
        P=28,
        Q=28,
        N=64,
    ),
    dict(
        C=128,
        M=128,
        R=3,
        S=3,
        P=28,
        Q=28,
        N=64,
    ),
    dict(
        C=128,
        M=256,
        R=3,
        S=3,
        P=14,
        Q=14,
        N=64,
    ),
    dict(
        C=256,
        M=256,
        R=3,
        S=3,
        P=14,
        Q=14,
        N=64,
    ),
    dict(
        C=256,
        M=512,
        R=3,
        S=3,
        P=7,
        Q=7,
        N=64,
    ),
    dict(
        C=512,
        M=512,
        R=3,
        S=3,
        P=7,
        Q=7,
        N=64,
    ),
]

resnet_18_layers = {
    "l1_conv1": 1, 
    "l2_conv2_stride2": 1,
    "l3-l7_conv2_stride1": 5,
    "l8_conv3_stride2": 1,
    "l9-l15_conv3_stride1": 7,
    "l16_conv4_stride2": 1,
    "l17-l27_conv4_stride1": 11,
    "l28_conv5_stride2": 1,
    "l29-l33_conv5_stride1": 5
}

