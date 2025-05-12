from autorocks.utils.enum import ExtendedEnum


class NNITuner(ExtendedEnum):
    # NNI specific: see registered_algorithms.yaml in nniopt/runtime/default_config/
    TPE = "TPE"
    RANDOM = "Random"
    BOHB = "BOHB"  # Doesn't work well for our problems - requires many restarts
    GP = "GPTuner"
    EVOLUTIONARY = "Evolution"
    SMAC = "SMAC"
    PBT = "PBTTuner"
