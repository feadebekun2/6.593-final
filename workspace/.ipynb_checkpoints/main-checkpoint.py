import os
from architectures.architecture_constants import Architecture, GPUMemoryScale, PEsConfig, RackSize, base_config, resnet_18_layers
from architectures.architecture_strategy import WorkloadStrategy
from architecture_results.architecture_results import ArchitectureResults
from matplot_results_plotter.matplot_results_plotter import MatplotResultsPlotter
from results_constants import ResultKeys
from datetime import datetime
import pytz

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
            ResultKeys.ENERGY: list,
            ResultKeys.CYCLES: list,
            ResultKeys.THROUGHPUT: (int, float),
            ResultKeys.STAR_HOPS: (int, float),
            ResultKeys.RING_HOPS: (int, float),
            ResultKeys.STAR_HOP_ENERGY: (int, float),
            ResultKeys.RING_HOP_ENERGY: (int, float),
        }
        for key, expected_type in required_keys.items():
            assert key in results
            assert isinstance(results[key], expected_type)

    """
    Runs the linear combos of all of the different configs (parallel strat, num GPUS, and memory per GPU)
    And collects results for each in self.results
    """
    def run_all(self):

        est = pytz.timezone('US/Eastern')
        timestamp = datetime.now(est).strftime("%B %d, %Y %I:%M:%S %p EST")
        filename = f"results_{timestamp}.json"
        filedir = "persisted_results"

        
        for arch in Architecture:
            for rack in RackSize:
                for scale in GPUMemoryScale:
                    for peConfig in PEsConfig:
                        key = (arch, scale, rack, peConfig)

                        print("=" * 50)
                        print("Starting Timeloop with configuration:")
                        print(f"  - arch: {key[0]}")
                        print(f"  - scale: {key[1]}")
                        print(f"  - rack: {key[2]}")
                        print(f"  - peConfig: {key[3]}")
                        print("=" * 50)

                        
                        # Skip configurations already computed.
                        if key in self.results.data:
                            print(f"Skipping {arch.name}, {scale.name}, {rack.name}, {peConfig.value}.")
                            continue
    
                        architecture = WorkloadStrategy(arch, scale, peConfig.value, rack.value,  self.debug)
                        results = architecture.run_workload()
    
                        self.__check_returned_vals(results)
    
                        self.results.add(
                            arch,
                            scale,
                            rack,
                            peConfig,
                            cycles=results[ResultKeys.CYCLES],
                            energy=results[ResultKeys.ENERGY],
                            tp=results[ResultKeys.THROUGHPUT],
                            star_hops=results[ResultKeys.STAR_HOPS],
                            ring_hops=results[ResultKeys.RING_HOPS],
                            star_hop_energy=results[ResultKeys.STAR_HOP_ENERGY],
                            ring_hop_energy=results[ResultKeys.RING_HOP_ENERGY],
                        )
    
                        # Ensure we save to persistent file to skip computations in the future
                        self.results.save_to_json(filename, filedir)
        


    def run_arch(self, arch: Architecture, filename: str, filedir: str):
        
        for rack in RackSize:
                for scale in GPUMemoryScale:
                    for peConfig in PEsConfig:
                        key = (arch, scale, rack, peConfig)

                        print("=" * 50)
                        print("Starting Timeloop with configuration:")
                        print(f"  - arch: {key[0]}")
                        print(f"  - scale: {key[1]}")
                        print(f"  - rack: {key[2]}")
                        print(f"  - peConfig: {key[3]}")
                        print("=" * 50)

                        
                        # Skip configurations already computed.
                        if key in self.results.data:
                            print(f"Skipping {arch.name}, {scale.name}, {rack.name}, {peConfig.value}.")
                            continue
    
                        architecture = WorkloadStrategy(arch, scale, peConfig.value, rack.value,  self.debug)
                        results = architecture.run_workload()
    
                        self.__check_returned_vals(results)
    
                        self.results.add(
                            arch,
                            scale,
                            rack,
                            peConfig,
                            cycles=results[ResultKeys.CYCLES],
                            energy=results[ResultKeys.ENERGY],
                            tp=results[ResultKeys.THROUGHPUT],
                            star_hops=results[ResultKeys.STAR_HOPS],
                            ring_hops=results[ResultKeys.RING_HOPS],
                            star_hop_energy=results[ResultKeys.STAR_HOP_ENERGY],
                            ring_hop_energy=results[ResultKeys.RING_HOP_ENERGY],
                        )
    
                        # Ensure we save to persistent file to skip computations in the future
                        self.results.save_to_json(filename, filedir)
        


        
    def from_results(filename, filedir):
        if os.path.exists(results_path):
            print("Loading in existing json file")
            self.results.load_from_json(filename, filedir)
        else:
            print("No json file found")

    
    """
    Plots the results stored in self.results
    Currently just iterates through each metric and will put the 3 strats on a single chart
    """
    def plot_results(self):
        for scale in GPUMemoryScale:
            for rack in RackSize:
                for metric in [ResultKeys.ENERGY, ResultKeys.CYCLES, ResultKeys.THROUGHPUT]:
                    # Collect results for this scale/rack across all architectures
                    results_dict = {}
                    for arch in Architecture:
                        value = getattr(self.results, f"get_{metric}")(arch, scale, rack)
                        # Sum lists to aggregate data
                        if isinstance(value, list):
                            value = sum(value)
                    
                        # Clamp any negative values to 0 (-1 represents an errored config)
                        if value is None or value < 0:
                            value = 0
                            
                        results_dict[arch] = value
    
                    if results_dict:
                        self.plotter.plot_bar_chart(
                            metric_key=metric,
                            results_dict=results_dict,
                            scale=scale,
                            rack=rack
                        )

