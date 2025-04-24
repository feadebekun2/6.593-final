from architectures.architecture_constants import Architecture, GPUMemoryScale, base_config, resnet_18_layers
from loaders import *

class WorkloadStrategy:
    def __init__(self, strategy: Architecture, gpu_architecture: GPUMemoryScale, num_gpus, debug):
        self.strategy = strategy
        self.gpu_architecture = gpu_architecture
        self.num_gpus = num_gpus
        self.debug = debug

        match self.strategy:
            case Architecture.Base:
                self.workload = self.__build_base_architecture()
            case Architecture.Data_Parallel:
                self.workload = self.__built_dp_architecture()
            case Architecture.Tensor_Parallel:
                self.workload = self.__build_tp_architecture()
            case _:
                raise ValueError(f"Unsupported Architecture: {strategy}")

    def __build_base_architecture(self):
        return base_config

    def __built_dp_architecture(self):
        dp_configs = []
        for config in base_config:
            dp_config = config.copy()
            dp_config["PE_spatial_factor_N"] *= self.num_gpus
            dp_config["global_buffer_factor_N"] *= self.num_gpus
            dp_config["DRAM_factor_N"] = int(dp_config["DRAM_factor_N"] / (self.num_gpus ** 2))
            dp_configs.append(dp_config)
        return dp_configs

    def __build_tp_architecture(self):
        tp_configs = []
        for config in base_config:
            tp_config = config.copy()
            tp_config["PE_spatial_factor_M"] *= self.num_gpus
            tp_config["global_buffer_factor_M"] *= self.num_gpus
            tp_config["DRAM_factor_M"] = int(tp_config["DRAM_factor_M"] / (self.num_gpus ** 2))
            tp_configs.append(tp_config)
        return tp_configs

    def run_workload(self):
        results = {"energy": [], "cycles": []}

        for i, (filename, count) in enumerate(resnet_18_layers.items()):
            config = self.workload[i]

            for _ in range(count):  # Repeat for repeated layers    
                # If debug is enabled, just add -1 to run through each config
                if self.debug:
                    results["energy"].append(-1)
                    results["cycles"].append(-1)
    
                # If debug is disabled, run the actual config
                else:
                    try:
                        result = run_timeloop_model(
                            config,
                            architecture=self.gpu_architecture.value,
                            mapping='designs/system/map.yaml',
                            problem=f"layer_shapes/{filename}.yaml",
                        )
            
                        with open('./output_dir/timeloop-model.stats.txt', 'r') as f:
                            stats = f.read()
            
                        lines = stats.split('\n')
                        energy = float([l for l in lines if 'Energy:' in l][0].split(' ', 2)[1])
                        cycles = int([l for l in lines if 'Cycles:' in l][0].split(' ', 1)[1])
        
                        results["energy"].append(energy)
                        results["cycles"].append(cycles)
        
                    except Exception as e:
                        print(f"Error running layer {filename}: {e}")
                        results["energy"].append(-1)
                        results["cycles"].append(-1)

        return results
        

    
