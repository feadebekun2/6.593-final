from architectures.architecture_constants import Architecture, GPUMemoryScale, base_config, resnet_18_layers, layer_shapes
from loaders import *
from enum import Enum

FREQUENCY = 1e9 # TODO: Find a better value (rn using 1GHz)

# From Timeloop output for DRAM
# Energy (per-scalar-access)               : 128.00 pJ
ENERGY_PER_HOP = 128 * 1e-6
NV_LINK_BANDWIDTH = 2.5e+10 #25 GB/s
BISECTION_BANDWIDTH = 10e+10 #Constant at 10 GB/s

class NetworkArch(Enum):
    STAR = "Star"
    RING = "Ring"

# conv2_stats = open('./output_dir/timeloop-model.stats.txt', 'r').read()

class DerivedMetricsEvaluator:
    def __init__(self, strategy: Architecture, gpu_architecture: GPUMemoryScale, num_gpus, workload):
        self.strategy = strategy
        self.gpu_architecture = gpu_architecture
        self.num_gpus = num_gpus
        self.workload = workload


    def parse_total_data_movement(stats):
        levels = re.split(r'Level \d+', text)[1:]  # Split by Level but skip the header
        total_bits_moved = 0
    
        for level_text in levels:
            # Extract Word bits
            word_bits_match = re.search(r'Word bits\s*:\s*(\d+)', level_text)
            word_bits = int(word_bits_match.group(1)) if word_bits_match else 16  # default 16 bits
    
            # Find reads, fills, updates
            scalar_reads = sum(int(x) for x in re.findall(r'Scalar reads \(per-instance\)\s*:\s*(\d+)', level_text))
            scalar_fills = sum(int(x) for x in re.findall(r'Scalar fills \(per-instance\)\s*:\s*(\d+)', level_text))
            scalar_updates = sum(int(x) for x in re.findall(r'Scalar updates \(per-instance\)\s*:\s*(\d+)', level_text))
    
            # Total words moved = reads + fills + updates
            words_moved = scalar_reads + scalar_fills + scalar_updates
    
            # Convert to bits
            bits_moved = words_moved * word_bits
            total_bits_moved += bits_moved
    
        # Convert total bits to bytes
        total_bytes_moved = total_bits_moved / 8
        return total_bytes_moved

    
    def derive_network_latency(self, stats, network_arch: NetworkArch):
        strategy_name = self.strategy.name
        
        # TODO: We need to account for the cost of a hop in terms of cycles

        return parse_total_data_movement(stats) / self.derive_link_bandwidth(network_arch)

    def derive_timeloop_latency(self, cycles, network_arch: NetworkArch):
        return cycles / self.derive_link_bandwidth(network_arch)

    def derive_link_bandwidth(self, network_arch: NetworkArch):
        if network_arch == "STAR":
            return BISECTION_BANDWIDTH / (self.num_gpus // 2)
        else if network_arch == "RING":
            return BISECTION_BANDWIDTH / 2
    
    """
    This is saying "how many times do the GPUs want to communicate with each other"
    This is not total network hops though because each time each GPU wants to communicate,
    the data might go through 1+ hops depending on the network topology
    """
    def __derive_total_communication_events(self):
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
            pe_N = self.workload[0].get('PE_spatial_factor_N', 1)

            return M * C * R * S * (pe_N - 1)
        elif strategy_name == "Tensor_Parallel":
            # TODO: Need to aggregate over each layer (but might need to multiply by 2 for forward and backward pass?)
            tot_hops = 0
            for layer in layer_shapes:
                C = layer["C"]
                M = layer["M"]
                R = layer["R"]
                S = layer["S"]
                pe_M = self.workload[0].get('PE_spatial_factor_M', 1)

                tot_hops += M * C * R * S * (pe_M - 1)
            return tot_hops
        else:
            raise ValueError(f"Unsupported Architecture: {self.strategy}")

    """
    https://www.geeksforgeeks.org/difference-between-star-and-ring-topology/
    A star network topology has a Hub in the middle and one link to each GPU

        Node 1        Node 2
               \     /
                 Hub
                /   \
         Node 3       Node 4

    The important thing to note, is that each GPU is always 2 hops away from any other GPU
    regardless of number of GPUs in the network
    """
    def derive_total_star_hops_and_energy(self):
        total_comms = self.__derive_total_communication_events()
        avg_hops_per_comm = 2

        total_hops = avg_hops_per_comm * total_comms
        return (total_hops, total_hops * ENERGY_PER_HOP)

    """
    https://www.geeksforgeeks.org/difference-between-star-and-ring-topology/
    A ring network topology has no hub and each GPU is connected to two adjacent GPUs
    in a circle format

                    Node 1
                  /        \
            Node 2          Node 3
                 \         /
                    Node 4

    The important thing here is that as n -> inf, avg_hops approaches num_gpus / 4 to account for longest path is half way around ring
    And then the average path is half the distance of half away around the ring.

    But we are only doing 2, 4, and 8 gpus.
    2 GPUs means each comm is always 1 away
    4 GPUs means each comm has a 2/3 chance of being 1 hop away and 1/3 chance of being 2 hops away (1.33 hops on average per comm)
    8 GPUs means each comm has 2 gpus 1 hop away, 2 gpus 2 hops away, 2 gpus 3 hops away, and only 1 gpu 4 hops away
        This means (2 * 1 + 2 * 2 + 2 * 3 + 1 * 4) = 16 hops per 7 GPUs = 16/7 (~2.29 hops on average per comm)
    """
    def derive_total_ring_hops_and_energy(self):
        total_comms = self.__derive_total_communication_events()
        avg_hops_per_comm = 0
        if self.num_gpus == 2:
            avg_hops_per_comm = 1
        if self.num_gpus == 4:
            avg_hops_per_comm = 4/3
        if self.num_gpus == 8:
            avg_hops_per_comm = 16/7

        total_hops = avg_hops_per_comm * total_comms
        return (total_hops, total_hops * ENERGY_PER_HOP)

