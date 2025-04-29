from collections import defaultdict
from enum import Enum
from typing import List
import json
import os
from architectures.architecture_constants import Architecture, GPUMemoryScale, RackSize, PEsConfig
from results_constants import ResultKeys

class ArchitectureResults:
    def __init__(self):
        self.data = defaultdict(lambda: {
            ResultKeys.CYCLES: [],
            ResultKeys.ENERGY: [],
            ResultKeys.THROUGHPUT: None,
            ResultKeys.STAR_HOPS: None,
            ResultKeys.RING_HOPS: None,
            ResultKeys.STAR_HOP_ENERGY: None,
            ResultKeys.RING_HOP_ENERGY: None,
        })

    def save_to_json(self, filename: str = "results.json", dir_path: str = "persisted_results"):
        os.makedirs(dir_path, exist_ok=True)
        full_path = os.path.join(dir_path, filename)
    
        # Prepare your new data
        new_data = {
            str((arch.name, scale.name, rack.name, peConfig.name)): metrics
            for (arch, scale, rack, peConfig), metrics in self.data.items()
        }
    
        # Step 1: Try loading existing data if the file exists
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {}
        else:
            existing_data = {}
    
        # Step 2: Merge existing data with new data
        existing_data.update(new_data)  # new_data overwrites keys if they already exist
    
        # Step 3: Write the merged data back
        with open(full_path, 'w') as f:
            json.dump(existing_data, f, indent=2)

    def load_from_json(self, filename: str = "results.json", dir_path: str = "persisted_results"):
        full_path = os.path.join(dir_path, filename)

        with open(full_path, 'r') as f:
            loaded = json.load(f)

        self.data.clear()
        for key_str, metrics in loaded.items():
            arch_name, scale_name, rack_name = eval(key_str)
            key = (
                Architecture[arch_name],
                GPUMemoryScale[scale_name],
                RackSize[rack_name],
            )
            self.data[key] = metrics

    def add(self, arch: Architecture, scale: GPUMemoryScale, rack: RackSize, peConfig: PEsConfig,
            cycles: List[float], energy: List[float],
            tp: float, star_hops: float, ring_hops: float,
            star_hop_energy: float, ring_hop_energy: float):
        key = (arch, scale, rack, peConfig)
        self.data[key][ResultKeys.CYCLES] = cycles
        self.data[key][ResultKeys.ENERGY] = energy
        self.data[key][ResultKeys.THROUGHPUT] = tp
        self.data[key][ResultKeys.STAR_HOPS] = star_hops
        self.data[key][ResultKeys.RING_HOPS] = ring_hops
        self.data[key][ResultKeys.STAR_HOP_ENERGY] = star_hop_energy
        self.data[key][ResultKeys.RING_HOP_ENERGY] = ring_hop_energy

    def get_cycles(self, arch: Architecture, scale: GPUMemoryScale, rack: RackSize) -> float:
        return self.data[(arch, scale, rack)][ResultKeys.CYCLES]

    def get_energy(self, arch: Architecture, scale: GPUMemoryScale, rack: RackSize) -> float:
        return self.data[(arch, scale, rack)][ResultKeys.ENERGY]

    def get_tp(self, arch: Architecture, scale: GPUMemoryScale, rack: RackSize) -> float:
        return self.data[(arch, scale, rack)][ResultKeys.THROUGHPUT]

    def get_star_hops(self, arch: Architecture, scale: GPUMemoryScale, rack: RackSize) -> float:
        return self.data[(arch, scale, rack)][ResultKeys.STAR_HOPS]

    def get_ring_hops(self, arch: Architecture, scale: GPUMemoryScale, rack: RackSize) -> float:
        return self.data[(arch, scale, rack)][ResultKeys.RING_HOPS]

    def get_star_hop_energy(self, arch: Architecture, scale: GPUMemoryScale, rack: RackSize) -> float:
        return self.data[(arch, scale, rack)][ResultKeys.STAR_HOP_ENERGY]

    def get_ring_hop_energy(self, arch: Architecture, scale: GPUMemoryScale, rack: RackSize) -> float:
        return self.data[(arch, scale, rack)][ResultKeys.RING_HOP_ENERGY]

    def all_keys(self):
        return self.data.keys()