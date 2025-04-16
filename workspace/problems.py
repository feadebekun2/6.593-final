from loaders import *

ARCH_CONFIG = {'pe_meshX': 4, 'pe_meshY': 4}

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

# Base configs keep everything in DRAM and have global_buffer_factor set to 1
base_configs = [
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

# For DP workloads, divide the DRAM factor N by 8 and multiply the buffer factor by 8
# for N since we are sharding along the batch dimension
def make_dp_workload(configs):
    dp_configs = []
    for config in configs:
        dp_config = config.copy()
        dp_config["PE_spatial_factor_N"] *= 4
        dp_config["global_buffer_factor_N"] *= 8
        dp_config["DRAM_factor_N"] = int(dp_config["DRAM_factor_N"] / 32)
        dp_configs.append(dp_config)
    return dp_configs

# For TP workloads, divide the DRAM factor M by 8 and multiply the buffer factor by 8
# for M since we are sharding along the weight dimension
def make_tp_workload(configs):
    tp_configs = []
    for config in configs:
        tp_config = config.copy()
        tp_config["PE_spatial_factor_M"] *= 4
        tp_config["global_buffer_factor_M"] *= 8
        tp_config["DRAM_factor_M"] = int(tp_config["DRAM_factor_M"] / 32)
        tp_configs.append(tp_config)
    return tp_configs

dp_configs = make_dp_workload(base_configs)
tp_configs = make_tp_workload(base_configs)


#We try and model resnet_18 in timeloop. 
#In layer_shapes, there are shapes of each convolutional filter involved in resnet. 
#TO DO: Add the fully connected layer, as well as a way to measure data reuse between layers (via skip connections)
#Maybe add LoopTree for this? 
def resnet_18_timeloop_loop():
    layers = {
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

    config_types = {
        "base": base_configs,
        "dp": dp_configs,
        "tp": tp_configs,
    }

    results = {
        "base": {"energy": [], "cycles": []},
        "dp": {"energy": [], "cycles": []},
        "tp": {"energy": [], "cycles": []},
    }

    for i, (filename, num_layers) in enumerate(layers.items()):
        print(i, filename, num_layers)
        base_config = base_configs[i]
        dp_config = dp_configs[i]
        tp_config = tp_configs[i]

        for config_type, config_list in config_types.items():
            print(f"  Running config: {config_type}")
            
            config = config_list[i]
        
            result = run_timeloop_model(
                config,
                architecture='designs/system/arch.yaml',
                mapping='designs/system/map.yaml',
                problem=f"layer_shapes/{filename}.yaml",
            )
        
            stats = open('./output_dir/timeloop-model.stats.txt', 'r').read()
    
            # Parse energy and cycles
            lines = stats.split('\n')
            energy = float([l for l in lines if 'Energy:' in l][0].split(' ', 2)[1])
            cycles = int([l for l in lines if 'Cycles:' in l][0].split(' ', 1)[1])
            num_hops = next((int(l.split(':')[1].strip()) for l in lines if 'Num-hops' in l), None)
        
            mapping = result.mapping

            results[config_type]["energy"].append(energy)
            results[config_type]["cycles"].append(cycles)
        
    for config_type in results:
        print(f"{config_type.upper()} RESULTS:")
        print("Energy:", results[config_type]["energy"])
        print("Cycles:", results[config_type]["cycles"])
    return results


