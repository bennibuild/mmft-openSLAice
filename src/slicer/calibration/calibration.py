from src.slicer.core.printer import Printer
from src.slicer.core.resin import CalibrationResin
from src.slicer.core.slicer import ParameterMode, Slicer
from src.slicer.core.constants import *
import os
import numpy as np

def create_calibration_print(
    printer: Printer,
    resin: CalibrationResin,
    export_path: str,
    layer_height: float = 0.05,
    exp_time_base: float = 4.0,
    exp_times_to_test: list[float] = [4.0, 3.7, 3.4, 3.1, 2.8, 2.5, 2.1, 1.8],
    exp_time_first_layers: float = 46.0,
    first_layers: int = 5
) -> None:
    """Create a calibration print with the given parameters."""
    
    first_layers_exp_times: list[float] = [exp_time_first_layers] * first_layers
    transition_exp_times: list[float] = np.linspace(exp_time_first_layers, exp_time_base, first_layers + 2)[1:-1].tolist() # type: ignore

    slicer_instance = Slicer(printer=printer, resin=resin)
    base_path = os.path.dirname(__file__)
    for i in range(1, 9):
        path = os.path.join(base_path, CAL_STL_FILES.replace("#", str(i)))
        slicer_instance.add_input_file(path)
    
    slicer_instance.auto_arrange(2)
    slicer_instance.set_layer_height_method(ParameterMode.MANUAL)
    slicer_instance.set_forced_layer_height(layer_height)
    slicer_instance.slice_all()
    slicer_instance.rasterize(255)

    layer_count = len(slicer_instance.layers_image)
    base_exposure_times = [exp_time_base] * (layer_count - (first_layers + first_layers + len(exp_times_to_test)))
    resin.set_exp_time(
        list(first_layers_exp_times)
        + list(transition_exp_times)
        + list(base_exposure_times)
        + list(exp_times_to_test)
    )

    slicer_instance.save(f"exp_cali_{exp_time_base}_{exp_times_to_test}", export_path)