from dataclasses import dataclass
import numpy as np

@dataclass
class ResinProperties:
    """
    A class to hold resin properties.

    Attributes
    ----------
    viscosity : float
        the viscosity in cps
    density : float
        the density in g/cm^3
    currency : str
        the currency of the price (e.g. 'EUR', '$')
    price : float
        the price in currency/L
    """
    viscosity: float
    density: float
    currency: str
    price: float

@dataclass
class ExposureSettings:
    """
    A class to hold resin settings.

    Attributes
    ----------
    light_off_delay : float
        the delay before turning off the light in seconds
    bottom_layers : int
        the number of bottom layers
    bottom_exp_time : float
        the exposure time for the bottom layers in seconds
    transition_layers : int
        the number of transition layers
    h_a : float
        the penetration depth in um
    T_c : float
        the time critical in sec
    overlap : float
        the overlap in mm, which is added to the layer height for the exposure time calculation
    """
    light_off_delay: float
    bottom_layers: int
    bottom_exp_time: float
    transition_layers: int
    h_a: float
    T_c: float
    overlap: float

@dataclass
class MovementSettings:
    """
    A class to hold movement settings.

    Attributes
    ----------
    bottom_lift_height : float
        the lift height for the bottom layers in mm
    bottom_lift_speed : float
        the lift speed for the bottom layers in mm/s
    bottom_retract_speed : float
        the retract speed for the bottom layers in mm/s
    normal_lift_height : float
        the lift height for normal layers in mm
    normal_lift_speed : float
        the lift speed for normal layers in mm/s
    normal_retract_speed : float
        the retract speed for normal layers in mm/s
    max_acceleration : float
        the maximum acceleration in mm/s^2
    use_TSMC: bool
        enables Two Stage Motion Control (TSMC) for the printer. If enabled the speeds are doubled for the second half of the lift and retract height.
    """
    bottom_lift_height: float
    bottom_lift_speed: float
    bottom_retract_speed: float
    normal_lift_height: float
    normal_lift_speed: float
    normal_retract_speed: float
    max_acceleration: float
    use_TSMC: bool

@dataclass
class SpecialSettings:
    """
    A class to hold special settings.

    Attributes
    ----------
    light_pwm : int
        the light pwm value between 0 and 255
    anti_aliasing : bool
        if anti aliasing is enabled
    min_aa : int
        the minimum anti aliasing value between 0 and 255 (255 = no aa, 0 = full aa)
    print_time_compensation : float
        the print time compensation in seconds
    shrinkage_compensation : float
        the shrinkage compensation in percent
    intelli_mode : bool
        if intelli mode is enabled
    """
    light_pwm: int
    anti_aliasing: int
    min_aa: int
    print_time_compensation: float
    shrinkage_compensation: float
    intelli_mode: bool


@dataclass(frozen=True)
class Resin:
    """
    A class to hold resin properties, settings, movement settings, and special settings.

    Attributes
    ----------
    name : str
        the name of the resin
    properties : ResinProperties
        the resin properties like viscosity, density, and price
    settings : ExposureSettings
        the resin settings like light off delay, bottom layers, and penetration curve values
    movement_settings : MovementSettings
        the movement settings like lift height, speed, and acceleration
    special_settings : SpecialSettings
        the special settings like light pwm, anti aliasing, and print time compensation
    """
    name: str
    properties: ResinProperties
    exposure_settings: ExposureSettings
    movement_settings: MovementSettings
    special_settings: SpecialSettings

    def get_exp_time(self, layer_index: int, layer_height: float) -> float:
        """
        Calculate the exposure time based on the layer height.
        The polymerization depth formula is: z_p = h_a × ln(t_p/T_c)
        The exposure time is calculated as: t_p = T_c × exp(z_p × (1 + overlap)/h_a)

        Parameters
        ----------
        layer_index : int
            The index of the layer.
        layer_height : float
            The height of the layer in mm.

        Returns
        -------
        float
            The exposure time in seconds.
        """
        if layer_index < self.exposure_settings.bottom_layers:
            return self.exposure_settings.bottom_exp_time

        z_p = (layer_height + self.exposure_settings.overlap) * 1000   # + overlap -> convert mm to um 
        calc_exp_time = self.exposure_settings.T_c * np.exp(z_p / self.exposure_settings.h_a)

        if layer_index < self.exposure_settings.bottom_layers + self.exposure_settings.transition_layers:
            # linear interpolation between bottom and normal exposure time
            exp_step = (self.exposure_settings.bottom_exp_time - calc_exp_time) / self.exposure_settings.transition_layers
            calc_exp_time += exp_step * (self.exposure_settings.transition_layers - (layer_index - self.exposure_settings.bottom_layers))
        return round(calc_exp_time, 2)
    

    def get_layer_time(self, layer_index: int, layer_height: float) -> float:
        """
        Calculate the total time for a layer including lift, exposure, and light off delay.
        The total time is calculated as:
        total_time = lift_up_time + lift_down_time + exposure_time + light_off_delay

        Parameters
        ----------
        layer_index : int
            The index of the layer.
        layer_height : float
            The height of the layer in mm.

        Returns
        -------
        float
            The total time for the layer in seconds.
        """
        n_move_up_time = self.movement_settings.normal_lift_height / self.movement_settings.normal_lift_speed if layer_index > self.exposure_settings.bottom_layers else self.movement_settings.bottom_lift_height / self.movement_settings.bottom_lift_speed
        n_move_down_time = self.movement_settings.normal_lift_height / self.movement_settings.normal_retract_speed if layer_index > self.exposure_settings.bottom_layers else self.movement_settings.bottom_lift_height / self.movement_settings.bottom_retract_speed
        return n_move_up_time + n_move_down_time + self.get_exp_time(layer_index, layer_height) + self.exposure_settings.light_off_delay


class CalibrationResin(Resin):
    def __init__(self, name, properties, exposure_settings, movement_settings, special_settings, exp_time: list[float]):
        super().__init__(name, properties, exposure_settings, movement_settings, special_settings)
        self.exp_time: list[float] = exp_time

    def set_exp_time(self, exp_time: list[float]):
        self.exp_time = exp_time + [1.]

    def get_exp_time(self, layer_index: int, layer_height: float):
        if layer_index < len(self.exp_time):
            return self.exp_time[layer_index]
        else:
            raise IndexError(f"Layer index {layer_index} exceeds the number of exposure times defined for this resin.")


class ResinRegistry:
    _resins: dict[str, Resin] = {}

    @classmethod
    def register(cls, resin: Resin):
        cls._resins[resin.name] = resin

    @classmethod
    def get(cls, name: str) -> Resin:
        resin = cls._resins.get(name)
        if resin is None:
            raise ValueError(f"Resin '{name}' is not registered.")
        return resin

    @classmethod
    def list_resins(cls):
        return list(cls._resins.values())
