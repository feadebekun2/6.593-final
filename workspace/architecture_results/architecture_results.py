from collections import defaultdict
from enum import Enum
from typing import List
from architectures.architecture_constants import Architecture, GPUMemoryScale, RackSize

class ArchitectureResults:
    def __init__(self):
        self.data = defaultdict(lambda: {"cycles": [], "energy": []})

    def add(self, arch: Architecture, scale: GPUMemoryScale, rack: RackSize, cycles: List[float], energy: List[float]):
        self.data[(arch, scale, rack)]["cycles"] = cycles
        self.data[(arch, scale, rack)]["energy"] = energy

    def get_cycles(self, arch: Architecture, scale: GPUMemoryScale, rack: RackSize) -> List[float]:
        return self.data[(arch, scale, rack)]["cycles"]

    def get_energy(self, arch: Architecture, scale: GPUMemoryScale, rack: RackSize) -> List[float]:
        return self.data[(arch, scale, rack)]["energy"]

    def all_keys(self):
        return self.data.keys()

    