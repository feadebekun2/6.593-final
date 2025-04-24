from collections import defaultdict
from enum import Enum
from typing import List
import json
import os
from architectures.architecture_constants import Architecture, GPUMemoryScale, RackSize

class ArchitectureResults:
    def __init__(self):
        self.data = defaultdict(lambda: {
            "cycles": [],
            "energy": [],
            "tp": None,
            "tot_hops": None,
            "hop_energy": None,
        })

    def save_to_json(self, filename: str = "results.json", dir_path: str = "persisted_results"):
        os.makedirs(dir_path, exist_ok=True)
        full_path = os.path.join(dir_path, filename)

        serializable_data = {
            str((arch.name, scale.name, rack.name)): metrics
            for (arch, scale, rack), metrics in self.data.items()
        }
        with open(full_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)

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

    def add(self, arch: Architecture, scale: GPUMemoryScale, rack: RackSize,
            cycles: List[float], energy: List[float],
            tp: float, tot_hops: float, hop_energy: float):
        key = (arch, scale, rack)
        self.data[key]["cycles"] = cycles
        self.data[key]["energy"] = energy
        self.data[key]["tp"] = tp
        self.data[key]["tot_hops"] = tot_hops
        self.data[key]["hop_energy"] = hop_energy

    def get_cycles(self, arch: Architecture, scale: GPUMemoryScale, rack: RackSize) -> List[float]:
        return self.data[(arch, scale, rack)]["cycles"]

    def get_energy(self, arch: Architecture, scale: GPUMemoryScale, rack: RackSize) -> List[float]:
        return self.data[(arch, scale, rack)]["energy"]

    def get_tp(self, arch: Architecture, scale: GPUMemoryScale, rack: RackSize) -> float:
        return self.data[(arch, scale, rack)]["tp"]

    def get_total_hops(self, arch: Architecture, scale: GPUMemoryScale, rack: RackSize) -> float:
        return self.data[(arch, scale, rack)]["tot_hops"]

    def get_hop_energy(self, arch: Architecture, scale: GPUMemoryScale, rack: RackSize) -> float:
        return self.data[(arch, scale, rack)]["hop_energy"]

    def all_keys(self):
        return self.data.keys()