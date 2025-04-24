from architectures.architecture_constants import Architecture, GPUMemoryScale, RackSize, base_config, resnet_18_layers
from architectures.architecture_strategy import WorkloadStrategy
from architecture_results.architecture_results import ArchitectureResults

class TimeLoopExperimentController:
    def __init__(self, debug=False):
        self.debug = debug
        self.results = ArchitectureResults()

    def run_all(self):
        for arch in Architecture:
            for rack in RackSize:
                for scale in GPUMemoryScale:
                    print(f"Running {arch} workload with {rack.value} GPUs on a {scale} memory size")
                    architecture = WorkloadStrategy(arch, scale, rack.value, self.debug)
                    results = architecture.run_workload()

                    assert len(results["cycles"]) + 1 == sum(resnet_18_layers.values())
                    
                    self.results.add(arch, scale, rack, results["cycles"], results["energy"])


    



