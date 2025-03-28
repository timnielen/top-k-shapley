from .CMCS import *
from .GapE import GapE
from .SAR import SAR
from .ApproShapley import ApproShapley
from .BUS import BUS
from .SVARM import SVARM, StratSVARM
from .KernelSHAP import KernelSHAP
from .shap_k import SHAP_K
from .compshap import compSHAP

algorithms = {
    "CMCS": CMCS,
    "CMCS@K": CMCS_at_K,
    "Greedy CMCS": Greedy_CMCS,
    "Identical": CMCS_Dependent,
    "Independent": CMCS_Independent,
    "Same Length": CMCS_Length,
    "GapE": GapE,
    "SAR": SAR,
    "ApproShapley": ApproShapley,
    "BUS": BUS,
    "SVARM": SVARM,
    "StratSVARM": StratSVARM,
    "KernelSHAP": KernelSHAP,
    "SamplingSHAP@K": SHAP_K,
    "compSHAP": compSHAP,
}