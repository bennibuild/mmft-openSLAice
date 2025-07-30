from src.slicer.core.printer import Printer, PrinterSettings, PrinterRegistry


settings = PrinterSettings(
    screen_resolution=(9024, 5120),
    pixel_size=0.017,
    max_z_height=165,
    z_resolution=0.001,
    max_z_speed=100,        # TODO find correct value
    max_z_acceleration=1000
)
anycubic_mono4_ultra = Printer(
    name="Anycubic Mono4 Ultra", 
    export_file_type="pm4u",
    settings=settings,
    min_layer_height=0.01,
    max_layer_height=0.10
)

PrinterRegistry.register(anycubic_mono4_ultra)