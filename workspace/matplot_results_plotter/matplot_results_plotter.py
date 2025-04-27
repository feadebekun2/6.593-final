import matplotlib.pyplot as plt
from architectures.architecture_constants import Architecture, GPUMemoryScale, RackSize, base_config, resnet_18_layers
from results_constants import ResultKeys

class MatplotResultsPlotter:
    def __init__(self):
        pass

    def __get_bar_label(self, arch: Architecture) -> str:
        return arch.name

    def __get_title(self, num_gpus: int, scale: GPUMemoryScale, add_on = '') -> str:
        plural = 's' if num_gpus > 1 else ''
        base_title = f"{num_gpus} GPU{plural} with {scale.size_label} memory"
        return f"{base_title} {add_on}".strip()

    def __get_ylabel(self, metric_key: str) -> str:
        units = {
            ResultKeys.CYCLES: " (cycles)",
            ResultKeys.ENERGY: " (pJ)",
            ResultKeys.THROUGHPUT: " (throughput, GOPS)",
            ResultKeys.STAR_HOPS: " (hops)",
            ResultKeys.RING_HOPS: " (hops)",
            ResultKeys.STAR_HOP_ENERGY: " (pJ)",
            ResultKeys.RING_HOP_ENERGY: " (pJ)",
        }
        return metric_key + units.get(metric_key, "")

    def plot_bar_chart(self, metric_key: str, results_dict: dict, scale: GPUMemoryScale, rack: RackSize):
        arch_names = [self.__get_bar_label(arch) for arch in results_dict.keys()]
        values = [results_dict[arch] for arch in results_dict.keys()]
        title = self.__get_title(rack.value, scale)
        ylabel = self.__get_ylabel(metric_key)

        plt.figure(figsize=(8, 5))
        plt.bar(arch_names, values, color='skyblue')
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel("Architecture")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()
    
    