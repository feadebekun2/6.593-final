from architectures.architecture_constants import Architecture, GPUMemoryScale, NetworkArch, PEsConfig, base_config, resnet_18_layers, layer_shapes
from loaders import *
from enum import Enum
import os
import re
import math

FREQUENCY = 1e9 # TODO: Find a better value (rn using 1GHz)

# From Timeloop output for DRAM
# Energy (per-scalar-access)               : 128.00 pJ
HOP_ENERGY_PER_BYTE = (128 * 1e-6) / 2 #pj to uJ
#DRAM is 128 pj / 16 bits, which its 64 pj / 1 byte, convert to microjoules. 

NV_LINK_BANDWIDTH = 2.5e+10 #25 GB/s
BISECTION_BANDWIDTH = 1e+10 #Constant at 10 GB/s or 1e10 B/s
CYCLES_PER_HOP = 80

# conv2_stats = open('./output_dir/timeloop-model.stats.txt', 'r').read()

class DerivedMetricsEvaluator:
    def __init__(self, strategy: Architecture, gpu_architecture: GPUMemoryScale, num_gpus, pe_config: PEsConfig, results_dir: str):
        self.strategy = strategy
        self.gpu_architecture = gpu_architecture
        self.num_gpus = num_gpus
        self.pe_config = pe_config
        
        config_str = f"{strategy.name}, {gpu_architecture.name}, {num_gpus.name}, {pe_config.name}"
        self.config_dir = os.path.join(results_dir, config_str)

        print(f"DerivedMetricsEvaluator created for {config_str}")

    def get_onchip_results(self):
        total_cycles = 0
        total_energy = 0
        
        
        layer_dir = os.path.join(self.config_dir, "Layer Results")

        
        for folder_name in os.listdir(layer_dir):
            
            #traverse through each layer. 
            folder_path = os.path.join(layer_dir, folder_name)
            
            if os.path.isdir(folder_path):
                # print(f"Found folder: {folder_name}")
                # You can process the folder here

                #get stats
                stats_file = os.path.join(folder_path, "stats.txt")

                if os.path.exists(stats_file):
                    with open(stats_file, "r") as f:
                        stats = f.read()
                        #parse stats. 
                        
                        lines = stats.split('\n')
                        energy = float([l for l in lines if 'Energy:' in l][0].split(' ', 2)[1]) * resnet_18_layers[folder_name]
                        cycles = int([l for l in lines if 'Cycles:' in l][0].split(' ', 1)[1]) * resnet_18_layers[folder_name]
                        total_cycles += cycles
                        total_energy += energy

                else:
                    print(f"No stats.txt in {folder_path}")
            else:
                print(f"Did not find folder for {folder_name}")
        return {'total_cycles': total_cycles, 'total_energy': total_energy}
               
        
               
        

    
        
    def parse_data_movement(self, layer_stats):
        returned = {'total_onchip_bytes': 0, 'total_DRAM_bytes': 0, 'total_bytes': 0}
                    
        levels = re.split(r'Level \d+', layer_stats)[1:]  # Split by Level but skip the header
        total_bits_moved = 0
        total_DRAM_bytes = 0 # this is just dram summed
        
        for idx, level_text in enumerate(levels):   
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

            # Network vs on-chip differentiation. 
            if idx == len(levels)-1:
                
                total_DRAM_bytes = bits_moved / 8

            
    
        # Convert total bits to bytes
        total_bytes_moved = total_bits_moved / 8

        return {
            'total_onchip_bytes': total_bytes_moved - total_DRAM_bytes, 
            'total_DRAM_bytes': total_DRAM_bytes, 
            'total_bytes': total_bytes_moved
           }

    

    def get_total_data_movement(self):

        layer_dir = os.path.join(self.config_dir, "Layer Results")
        
        returned = {'total_network_bytes': [], 'total_onchip_bytes': []}

        folders = os.listdir(layer_dir)

        
        #for TP, Base: each layer needs to be moved.
        folders_to_iterate = []
        for layer_name in folders:
            layer_count = resnet_18_layers[layer_name]
            folders_to_iterate.extend([layer_name] * layer_count)

        
        # #for DP: network is 0th and last.
        # if self.strategy == Architecture.Data_Parallel:
        #     folders_to_iterate = [folders[0], folders[-1]]
        
        for idx, folder_name in enumerate(folders_to_iterate):
            
            #traverse through each layer. 
            folder_path = os.path.join(layer_dir, folder_name)
            
            if os.path.isdir(folder_path):
                # print(f"Found folder: {folder_name}")
                # You can process the folder here

                #get stats
                stats_file = os.path.join(folder_path, "stats.txt")

                if os.path.exists(stats_file):
                    with open(stats_file, "r") as f:
                        stats = f.read()
                        #parse stats. 
                        layer_result = self.parse_data_movement(stats)

                        #for base, all on chip. 
                        if self.strategy == Architecture.Base:
                            returned['total_onchip_bytes'].append(layer_result['total_bytes'])
                            returned['total_network_bytes'].append(0)
                                
                        elif self.strategy == Architecture.Data_Parallel:
                            if idx == 0:
                                returned['total_network_bytes'].append(layer_result['total_DRAM_bytes'])
                                returned['total_onchip_bytes'].append(layer_result['total_onchip_bytes'])
                            
                            elif idx == len(folders_to_iterate)-1:
                                returned['total_onchip_bytes'].append(layer_result['total_onchip_bytes'])
                                # last layer has M=512, P,Q = 7 and each number is 16 bits / a byte
                                # 64 total images, split among num_gpus GPUs. 
                                returned['total_network_bytes'].append(512 * 7 * 7 * (16 / 8) *64)
                            else:
                                returned['total_network_bytes'].append(0)
                                returned['total_onchip_bytes'].append(layer_result['total_onchip_bytes'])
                        elif self.strategy == Architecture.Tensor_Parallel:
                             returned['total_network_bytes'].append(layer_result['total_DRAM_bytes'])
                             returned['total_onchip_bytes'].append(layer_result['total_onchip_bytes'])
                        else:
                            raise Exception('unknown arch')
                    

                else:
                    print(f"No stats.txt in {folder_path}")
            else:
                print(f"Did not find folder for {folder_name}")

        return returned
        
    #We will use these to compare - is the output bottlenecked by network latency? Or timeloop latency? 

    
    def derive_network_latency(self, stats, network_arch: NetworkArch):
        strategy_name = self.strategy.name
        
        # TODO: We need to account for the cost of a hop in terms of cycles

        return parse_total_data_movement(stats) / self.derive_link_bandwidth(network_arch)

    def derive_timeloop_latency(self, cycles, network_arch: NetworkArch):
        return cycles / self.derive_link_bandwidth(network_arch)

    def derive_link_bandwidth(self, network_arch: NetworkArch):
        if network_arch == NetworkArch.STAR:
            return BISECTION_BANDWIDTH / (self.num_gpus.value // 2)
        elif network_arch == NetworkArch.RING:
            return BISECTION_BANDWIDTH / 2
    
   
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
    def derive_total_star_results(self):

        #{'total_network_bytes': 0, 'total_onchip_bytes': 0}
        total_bytes = self.get_total_data_movement()
        
        avg_hops_per_comm = 2

        link_bandwidth_in_cycles = self.derive_link_bandwidth(NetworkArch.STAR) / FREQUENCY
        
        #per layer, calculate total hops. 
        total_network_bytes_array = total_bytes['total_network_bytes']
       
        # for layer_num_bytes in total_network_bytes_array:
        #     bytes_per_gpu = layer_num_bytes / self.num_gpus.value
        #     sends = math.ceil(bytes_per_gpu/max_data_per_comm_event)
        #     total_sends += sends
        
        # total_network_hops = total_sends * avg_hops_per_comm
        total_network_hops = 0
        total_network_energy = 0
        network_latency = 0
        
        for layer_num_bytes in total_network_bytes_array:
            bytes_per_gpu = layer_num_bytes / self.num_gpus.value * (self.num_gpus.value - 1)
            
            # For energy: sum total byte-hop movements
            #this is total hops per 
            total_hops_for_layer = layer_num_bytes * avg_hops_per_comm
            total_network_hops += total_hops_for_layer
            total_network_energy += total_hops_for_layer * HOP_ENERGY_PER_BYTE
        
            # For latency: how long does it take for 1 GPU to send its bytes?
            comm_cycles = bytes_per_gpu / link_bandwidth_in_cycles
            per_gpu_latency = comm_cycles + avg_hops_per_comm * CYCLES_PER_HOP
            network_latency += per_gpu_latency

        
        # now divide the total data moved, scaled by hops, by the bandwidth that can be delivered for send
       
        
        # (total times this strategy sends data on the network, energy used for network sends, on chip data movement)
        on_chip_bytes = total_bytes['total_onchip_bytes']
        # return (total_network_sends, total_network_sends * ENERGY_PER_HOP, total_comms['total_onchip_bytes'] / on_chip_bandwidth)

        on_chip_results = self.get_onchip_results()

        on_chip_latency = on_chip_results['total_cycles']
        
        
        bottlenecked_latency = max(on_chip_latency, network_latency)
        
        return {
                'per_layer_network_bytes': total_bytes['total_network_bytes'],
                'per_layer_onchip_bytes': total_bytes['total_onchip_bytes'],
                'total_network_bytes': sum(total_bytes['total_network_bytes']),
                'total_network_hops': total_network_hops, 
                'total_network_latency': network_latency, 
                'total_network_energy': total_network_energy,
                'total_onchip_bytes': sum(total_bytes['total_onchip_bytes']),
                'total_onchip_latency': on_chip_latency, 
                'total_onchip_energy': on_chip_results['total_energy'],
                'bottlenecked_latency': bottlenecked_latency
       }
        # Roofline model is saying:
        # If total_network_sends * cycles per network send > on chip sends * cycles per on chip send, then we are network bound
        # otherwise we are on chip bound
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
    16 GPUs means same math as 8 which is:
        64 hops/ 15 nodes = ~4.266666...
    
    """
    def derive_total_ring_results(self):
        total_bytes = self.get_total_data_movement()
        print("seeing num_gpus: ", self.num_gpus.value)
        avg_hops_per_comm = 0
        if self.num_gpus.value == 2:
            avg_hops_per_comm = 1
        if self.num_gpus.value == 4:
            avg_hops_per_comm = 4/3
        if self.num_gpus.value == 8:
            avg_hops_per_comm = 16/7
        if self.num_gpus.value == 16:
            avg_hops_per_comm = 64/15

        link_bandwidth_in_cycles = self.derive_link_bandwidth(NetworkArch.RING) / FREQUENCY
        
        #per layer, calculate total hops. 
        total_network_bytes_array = total_bytes['total_network_bytes']
       
        # for layer_num_bytes in total_network_bytes_array:
        #     bytes_per_gpu = layer_num_bytes / self.num_gpus.value
        #     sends = math.ceil(bytes_per_gpu/max_data_per_comm_event)
        #     total_sends += sends
        
        # total_network_hops = total_sends * avg_hops_per_comm
        # total_network_hops = 0
        # total_network_energy = 0
        # for layer_num_bytes in total_network_bytes_array:
        #     bytes_per_gpu = layer_num_bytes / self.num_gpus.value
        #     #cycles to send those bytes. 
        #     comm_cycles = bytes_per_gpu / link_bandwidth_in_cycles #in cycles

        #     #total network hops for this layer
        #     network_hops = comm_cycles * avg_hops_per_comm #in hops. 

        #     total_network_energy += bytes_per_gpu *(HOP_ENERGY_PER_BYTE) * network_hops
        #     total_network_hops += network_hops


        total_network_hops = 0
        total_network_energy = 0
        network_latency = 0
        
        for layer_num_bytes in total_network_bytes_array:
            bytes_per_gpu = (layer_num_bytes / self.num_gpus.value) * (self.num_gpus.value - 1)
            
            # For energy: sum total byte-hop movements

            #
            total_hops_for_layer = layer_num_bytes * avg_hops_per_comm
            total_network_hops += total_hops_for_layer
            total_network_energy += total_hops_for_layer * HOP_ENERGY_PER_BYTE
        
            # For latency: how long does it take for 1 GPU to send its bytes?
            comm_cycles = bytes_per_gpu / link_bandwidth_in_cycles
            per_gpu_latency = comm_cycles + avg_hops_per_comm * CYCLES_PER_HOP
            network_latency += per_gpu_latency
        
        # now divide the total data moved, scaled by hops, by the bandwidth that can be delivered for send
       
        
        # (total times this strategy sends data on the network, energy used for network sends, on chip data movement)
        on_chip_bytes = total_bytes['total_onchip_bytes']
        # return (total_network_sends, total_network_sends * ENERGY_PER_HOP, total_comms['total_onchip_bytes'] / on_chip_bandwidth)

        on_chip_results = self.get_onchip_results()

        on_chip_latency = on_chip_results['total_cycles']
        # network_latency = total_network_hops * CYCLES_PER_HOP
        
        bottlenecked_latency = max(on_chip_latency, network_latency)
        
        return {
                'per_layer_network_bytes': total_bytes['total_network_bytes'],
                'per_layer_onchip_bytes': total_bytes['total_onchip_bytes'],
                'total_network_bytes': sum(total_bytes['total_network_bytes']),
                'total_network_hops': total_network_hops, 
                'total_network_latency': network_latency, 
                'total_network_energy': total_network_energy,
                'total_onchip_bytes': sum(total_bytes['total_onchip_bytes']),
                'total_onchip_latency': on_chip_latency, 
                'total_onchip_energy': on_chip_results['total_energy'],
                'bottlenecked_latency': bottlenecked_latency
       }

        

