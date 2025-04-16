from loaders import *


ARCH_CONFIG = {'pe_meshX': 4, 'pe_meshY': 4}
config = dict(
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
)

def resnet_18_timeloop_loop():

    layers = {"l1_conv1": 1, 
              "l2_conv2_stride2": 1,
              "l3-l7_conv2_stride1": 5,
              "l8_conv3_stride2": 1,
              "l9-l15_conv3_stride1": 7,
              "l16_conv4_stride2": 1,
              "l17-l27_conv4_stride1": 11,
              "l28_conv5_stride2": 1,
              "l29-l33_conv5_stride1": 5
     }
        
    # total-energy = 0
    # total-latency = 0
    # print(layers.items)
    print(f"CONFIG: {config}")
    energy_results = []
    cycle_results = []
    for filename, num_layers in layers.items():
        
        # print(filename, num_layers)
        for _ in range(num_layers):
            result = run_timeloop_model(
                config,
                architecture='designs/system/arch.yaml',
                mapping='designs/system/map.yaml',
                problem=f"layer_shapes/{filename}.yaml"
            )
            stats = open('./output_dir/timeloop-model.stats.txt', 'r').read()
        
            # Parse energy and cycles
            lines = stats.split('\n')
            energy = float([l for l in lines if 'Energy:' in l][0].split(' ', 2)[1])
            cycles = int([l for l in lines if 'Cycles:' in l][0].split(' ', 1)[1])
            num_hops = next((int(l.split(':')[1].strip()) for l in lines if 'Num-hops' in l), None)
        
            mapping = result.mapping
        
            energy_line = [l for l in lines if 'Energy:' in l][0]
            # print("Energy line:", energy_line)
        
            print(energy, cycles)
            # config_names_results.append(config_name)
            energy_results.append(energy)
            cycle_results.append(cycles)
            # throughput = total_MACs / cycles
            # throughput_results.append(throughput)
        
    print(energy_results)
    print(cycle_results)
    return


