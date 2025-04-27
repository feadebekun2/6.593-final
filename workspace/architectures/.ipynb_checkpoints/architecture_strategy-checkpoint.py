from architectures.architecture_constants import Architecture, GPUMemoryScale, base_config, resnet_18_layers
from architecture_results.derived_metrics_evaluator import DerivedMetricsEvaluator
from loaders import *
from results_constants import ResultKeys

class WorkloadStrategy:
    def __init__(self, strategy: Architecture, gpu_architecture: GPUMemoryScale, num_gpus, debug):
        self.strategy = strategy
        self.gpu_architecture = gpu_architecture
        self.num_gpus = num_gpus
        self.debug = debug

        if self.strategy == Architecture.Base:
            self.workload = self.__build_base_architecture()
        elif self.strategy == Architecture.Data_Parallel:
            self.workload = self.__built_dp_architecture()
        elif self.strategy == Architecture.Tensor_Parallel:
            self.workload = self.__build_tp_architecture()
        else:
            raise ValueError(f"Unsupported Architecture: {strategy}")

        self.derivedMetricsEvaluator = DerivedMetricsEvaluator(strategy, gpu_architecture, num_gpus, self.workload)

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
        results = {
            ResultKeys.ENERGY: [],
            ResultKeys.CYCLES: [],
            ResultKeys.THROUGHPUT: 0,
            ResultKeys.STAR_HOPS: 0,
            ResultKeys.RING_HOPS: 0,
            ResultKeys.STAR_HOP_ENERGY: 0,
            ResultKeys.RING_HOP_ENERGY: 0,
        }

        flat_index = 0
        for i, (filename, count) in enumerate(resnet_18_layers.items()):
            config = self.workload[i]

            for _ in range(count):  # Repeat for repeated layers
                print(f"Running layer {flat_index + 1}: {filename}")
                flat_index += 1
                # Skip computation on configs that already had errors
                if results[ResultKeys.THROUGHPUT] == -1:
                    results[ResultKeys.ENERGY].append(-1)
                    results[ResultKeys.CYCLES].append(-1)
                    results[ResultKeys.THROUGHPUT] = -1
                    continue

                # If debug is enabled, just add -1 to run through each config
                if self.debug:
                    results[ResultKeys.ENERGY].append(-1)
                    results[ResultKeys.CYCLES].append(-1)
                    results[ResultKeys.THROUGHPUT] = -1
    
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
        
                        results[ResultKeys.ENERGY] = energy
                        results[ResultKeys.CYCLES] = cycles
    
        
                    except Exception as e:
                        print(f"Error running layer {filename}: {e}")
                        results[ResultKeys.ENERGY].append(-1)
                        results[ResultKeys.CYCLES].append(-1)
                        results[ResultKeys.THROUGHPUT] = -1

        if results[ResultKeys.THROUGHPUT] != -1:
            results[ResultKeys.THROUGHPUT] = self.derivedMetricsEvaluator.derive_throughput(results[ResultKeys.CYCLES])
        
        # These are independent of failures
        star_hops, star_hop_energy = self.derivedMetricsEvaluator.derive_total_star_hops_and_energy()
        ring_hops, ring_hop_energy = self.derivedMetricsEvaluator.derive_total_ring_hops_and_energy()
        
        results[ResultKeys.STAR_HOPS] = star_hops
        results[ResultKeys.RING_HOPS] = ring_hops
        results[ResultKeys.STAR_HOP_ENERGY] = star_hop_energy
        results[ResultKeys.RING_HOP_ENERGY] = ring_hop_energy

        return results
        

    
