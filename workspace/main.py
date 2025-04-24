import os
from architectures.architecture_constants import Architecture, GPUMemoryScale, RackSize, base_config, resnet_18_layers
from architectures.architecture_strategy import WorkloadStrategy
from architecture_results.architecture_results import ArchitectureResults
from matplot_results_plotter.matplot_results_plotter import MatplotResultsPlotter

class TimeLoopExperimentController:
    def __init__(self, debug=False):
        self.debug = debug
        self.results = ArchitectureResults()
        self.plotter = MatplotResultsPlotter()

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
        results_path = "persisted_results/results.json"

        if os.path.exists(results_path):
            print("Loading in existing json file")
            self.results.load_from_json()
        else:
            print("No json file found")
        
        for arch in Architecture:
            for rack in RackSize:
                for scale in GPUMemoryScale:
                    key = (arch, scale, rack)

                    # Skip configurations already computed.
                    if key in self.results.data:
                        print(f"Skipping {arch.name}, {scale.name}, {rack.name}.")
                        continue

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

                    # Ensure we save to persistent file to skip computations in the future
                    self.results.save_to_json()

    """
    Plots the results stored in self.results
    Currently just iterates through each metric and will put the 3 strats on a single chart
    """
    def plot_results(self):
        for scale in GPUMemoryScale:
            for rack in RackSize:
                for metric in ["energy", "cycles", "tp", "hop_energy"]:
                    # Collect results for this scale/rack across all architectures
                    results_dict = {}
                    for arch in Architecture:
                        value = getattr(self.results, f"get_{metric}")(arch, scale, rack)
                        # If it's a list, take sum or average as needed
                        if isinstance(value, list):
                            value = sum(value)
                        results_dict[arch] = value
    
                    if results_dict:
                        self.plotter.plot_bar_chart(
                            metric_key=metric,
                            results_dict=results_dict,
                            scale=scale,
                            rack=rack
                        )

