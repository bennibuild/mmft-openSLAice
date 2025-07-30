from src.slicer.core.resin import Resin, ResinProperties, ExposureSettings, MovementSettings, SpecialSettings, ResinRegistry, CalibrationResin


properties = ResinProperties(
    viscosity=200,
    density=1.13,
    currency="EUR",
    price=40.00
)

exposure_settings = ExposureSettings(
    light_off_delay=2,
    bottom_layers=0,
    bottom_exp_time=0,
    transition_layers=0,
    h_a=0,
    T_c=0.,
    overlap=0.
)

movement_settings = MovementSettings(
    bottom_lift_height=8,
    bottom_lift_speed=1.5,
    bottom_retract_speed=3,
    normal_lift_height=6,
    normal_lift_speed=2,
    normal_retract_speed=3,
    max_acceleration=20,
    use_TSMC=False
)

special_settings = SpecialSettings(
    light_pwm=255,
    anti_aliasing=1,
    min_aa=255,
    print_time_compensation=0.0,
    shrinkage_compensation=0.0,
    intelli_mode=True
)

calibration_resin = CalibrationResin(
    name="Resin for Calibration",
    properties=properties,
    exposure_settings=exposure_settings,
    movement_settings=movement_settings,
    special_settings=special_settings,
    exp_time=[],
)

ResinRegistry.register(calibration_resin)