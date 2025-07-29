from dataclasses import dataclass
from typing import TypedDict

@dataclass
class PrinterSettings():
    """
    A class to hold printer settings.

    Attributes
    ----------
    screen_resolution : tuple[int, int]
        the screen resolution in pixels (width, height)
    pixel_size : float
        the pixel size in mm
    max_z_height : float
        the maximum layer height of the printer in mm
    z_resolution : float
        the vertical resolution of the printer in mm
    max_z_speed : float
        the maximum speed of the printer in mm/s
    max_z_acceleration : float
        the maximum acceleration of the printer in mm/s^2
    """
    screen_resolution: tuple[int, int]
    pixel_size: float
    max_z_height: float
    z_resolution: float
    max_z_speed: float
    max_z_acceleration: float


@dataclass(frozen=True)
class Printer:
    """
    A class to hold printer settings.

    Attributes
    ----------
    name : str
        the name of the printer
    export_file_type : str
        the file type the printer can export
    settings : PrinterSettings
        the settings of the printer
    min_layer_height: float
        the min layer height in mm the printer is able to print
    max_layer_height: float
        the max layer height in mm the printer is able to print
    """
    name: str
    export_file_type: str
    settings: PrinterSettings
    min_layer_height: float
    max_layer_height: float


class PrinterRegistry:
    """
    A class to manage printer registrations.

    Attributes
    ----------
    _printers : dict[str, Printer]
        a dictionary to hold all registered printers by name

    Methods
    ----------
    register(printer: Printer)
        register a printer
    get(name: str) -> Printer
        get a printer by name
    list_printers() -> list[Printer]
        get a list of all registered printers
    """
    _printers: dict[str, Printer] = {}

    @classmethod
    def register(cls, printer: Printer):
        cls._printers[printer.name] = printer

    @classmethod
    def get(cls, name: str) -> Printer:
        printer = cls._printers.get(name)
        if printer is None:
            raise ValueError(f"Printer '{name}' is not registered.")
        return printer

    @classmethod
    def list_printers(cls):
        return list(cls._printers.values())