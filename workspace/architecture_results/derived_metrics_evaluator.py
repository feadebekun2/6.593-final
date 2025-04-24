from architectures.architecture_constants import Architecture, GPUMemoryScale, base_config, resnet_18_layers, layer_shapes
from loaders import *

FREQUENCY = 1e9 # TODO: Find a better value (rn using 1GHz)
ENERGY_PER_HOP = 0.02 # TODO: Get both units of energy and also the value of a DRAM access to use as the estimated value

class DerivedMetricsEvaluator:
    def __init__(self, strategy: Architecture, gpu_architecture: GPUMemoryScale, num_gpus, workload):
        self.strategy = strategy
        self.gpu_architecture = gpu_architecture
        self.num_gpus = num_gpus
        self.workload = workload

    def derive_throughput(self, cycles):
        strategy_name = self.strategy.name
        
        # TODO: We need to account for the cost of a hop in terms of cycles
        if strategy_name == "Base":
            return cycles / FREQUENCY
        elif strategy_name == "Data_Parallel":
            return cycles / FREQUENCY
        elif strategy_name == "Tensor_Parallel":
            return cycles / FREQUENCY
        else:
            raise ValueError(f"Unsupported Architecture: {self.strategy}")

    def derive_total_hops(self):
        strategy_name = self.strategy.name
        
        if strategy_name == "Base":
            # No hops in the base architecture since all work is on 1 GPU
            return 0
        elif strategy_name == "Data_Parallel":
            # Only hop is at the very end so just use layer_shapes[0]
            C = layer_shapes[0]["C"]
            M = layer_shapes[0]["M"]
            R = layer_shapes[0]["R"]
            S = layer_shapes[0]["S"]
            pe_N = self.workload.get('PE_spatial_factor_N', 1)

            return M * C * R * S * (pe_N - 1)
        elif strategy_name == "Tensor_Parallel":
            # TODO: Need to aggregate over each layer (but might need to multiply by 2 for forward and backward pass?)
            tot_hops = 0
            for layer in layer_shapes:
                C = layer["C"]
                M = layer["M"]
                R = layer["R"]
                S = layer["S"]
                pe_M = self.workload.get('PE_spatial_factor_M', 1)

                tot_hops += M * C * R * S * (pe_M - 1)
            return tot_hops
        else:
            raise ValueError(f"Unsupported Architecture: {self.strategy}")

    def derive_total_hop_energy(self):
        return self.derive_total_hops() * ENERGY_PER_HOP