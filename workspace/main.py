from architectures.architecture_constants import Architecture, GPUMemoryScale, RackSize, base_config, resnet_18_layers
from architectures.architecture_strategy import WorkloadStrategy
from architecture_results.architecture_results import ArchitectureResults

class TimeLoopExperimentController:
    def __init__(self, debug=False):
        self.debug = debug
        self.results = ArchitectureResults()

    """
    Checks the integrity of the data returned.
    """
    def __check_returned_vals(self, results):
        # Make sure each layer was accounted for
        assert len(results["cycles"]) == sum(resnet_18_layers.values())

        # Make sure returned values is of correct shape
        required_keys = {
            "energy": list,
            "cycles": list,
            "tp": (int, float),
            "tot_hops": (int, float),
            "hop_energy": (int, float),
        }
        for key, expected_type in required_keys.items():
            assert key in results
            assert isinstance(results[key], expected_type)

    """
    Runs the linear combos of all of the different configs (parallel strat, num GPUS, and memory per GPU)
    And collects results for each in self.results
    """
    def run_all(self):
        for arch in Architecture:
            for rack in RackSize:
                for scale in GPUMemoryScale:
                    # print(f"Running {arch} workload with {rack.value} GPUs on a {scale} memory size")
                    architecture = WorkloadStrategy(arch, scale, rack.value, self.debug)
                    results = architecture.run_workload()

                    self.__check_returned_vals(results)

                    self.results.add(
                        arch,
                        scale,
                        rack,
                        cycles=results["cycles"],
                        energy=results["energy"],
                        tp=results["tp"],
                        tot_hops=results["tot_hops"],
                        hop_energy=results["hop_energy"]
                    )




    



