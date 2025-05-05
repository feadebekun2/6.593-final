from architectures.architecture_constants import Architecture, GPUMemoryScale, PEsConfig, base_config, resnet_18_layers
from architecture_results.derived_metrics_evaluator import DerivedMetricsEvaluator
from loaders import *
from results_constants import ResultKeys

class WorkloadStrategy:
    def __init__(self, strategy: Architecture, gpu_architecture: GPUMemoryScale, peConfig: PEsConfig, num_gpus, debug):
        self.strategy = strategy
        self.gpu_architecture = gpu_architecture
        self.num_gpus = num_gpus
        self.debug = debug
        self.peConfig = peConfig
       
        if self.strategy == Architecture.Base:
            self.constraints = 'designs/system/constraints_Base.yaml'
        elif self.strategy == Architecture.Data_Parallel:
            self.constraints = 'designs/system/constraints_DP.yaml'
        elif self.strategy == Architecture.Tensor_Parallel:
            self.constraints = 'designs/system/constraints_TP.yaml'
        else:
            raise ValueError(f"Unsupported Architecture: {strategy}")

    def run_workload(self):
        results = {
            ResultKeys.ENERGY: [],
            ResultKeys.CYCLES: [],
            ResultKeys.THROUGHPUT: 1,
            ResultKeys.STAR_HOPS: 0,
            ResultKeys.RING_HOPS: 0,
            ResultKeys.STAR_HOP_ENERGY: 0,
            ResultKeys.RING_HOP_ENERGY: 0,
        }

        layer_stats = {}
        layer_mappings = {}
        
        flat_index = 0
        for i, (filename, count) in enumerate(resnet_18_layers.items()):
            
            print(f"Running layer {flat_index + 1}: {filename} which occurs {count} times")

            
            # Skip computation on configs that already had errors
            if results[ResultKeys.THROUGHPUT] == -1:
                results[ResultKeys.ENERGY].append(-1)
                results[ResultKeys.CYCLES].append(-1)
                continue

            # If debug is enabled, just add -1 to run through each config
            if self.debug:
                results[ResultKeys.ENERGY].append(-1)
                results[ResultKeys.CYCLES].append(-1)
                results[ResultKeys.THROUGHPUT] = -1

            # If debug is disabled, run the actual config
            else:
                try:
                    result = run_timeloop_mapper(
                        {'num_gpus': self.num_gpus, 'pe_meshX': self.peConfig.pe_meshX, 'pe_meshY': self.peConfig.pe_meshY},
                        constraints=self.constraints,
                        architecture=self.gpu_architecture.value,
                        mapper='designs/_include/mapper.yaml',
                        problem=f"layer_shapes/{filename}.yaml",
                    )
                    

                    #get results, mapping, and stats. 
                    layer_mappings[filename] = result.mapping
                    with open('./output_dir/timeloop-mapper.stats.txt', 'r') as f:
                        stats = f.read()

                    layer_stats[filename] = stats
                    
                    lines = stats.split('\n')
                    energy = float([l for l in lines if 'Energy:' in l][0].split(' ', 2)[1])
                    cycles = int([l for l in lines if 'Cycles:' in l][0].split(' ', 1)[1])

                    for i in range(count):
                        print(f"Adding energy and cycle stats for {filename} occurence {i+1}")
                        results[ResultKeys.ENERGY].append(energy)
                        results[ResultKeys.CYCLES].append(cycles)
                        flat_index += 1

    
                except Exception as e:
                    print(f"Error running layer {filename}: {e}")
                    for i in range(count):
                        results[ResultKeys.ENERGY].append(-1)
                        results[ResultKeys.CYCLES].append(-1)
                    results[ResultKeys.THROUGHPUT] = -1
        
        # These are independent of failures
        star_hops, star_hop_energy = 0, 0
        ring_hops, ring_hop_energy = 0, 0
        
        results[ResultKeys.STAR_HOPS] = star_hops
        results[ResultKeys.RING_HOPS] = ring_hops
        results[ResultKeys.STAR_HOP_ENERGY] = star_hop_energy
        results[ResultKeys.RING_HOP_ENERGY] = ring_hop_energy

        for key in (ResultKeys.ENERGY, ResultKeys.CYCLES):
            arr = results[key]
            if -1 in arr:
                # calculate how many -1 we need to add
                deficit = 33 - len(arr)
                if deficit > 0:
                    arr.extend([-1] * deficit)
        print(results)

        return results, layer_stats, layer_mappings
        

    
