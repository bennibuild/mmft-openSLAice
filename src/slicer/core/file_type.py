from abc import ABC, abstractmethod
from typing import Protocol, List
from PIL.Image import Image
from src.slicer.core.printer import Printer
from src.slicer.core.resin import Resin


class FileType(Protocol):
    """
    A protocol for file types used in the slicer.
    This protocol defines the structure that all file types must adhere to.
    """
    type_name: str
    
    @abstractmethod
    def __init__(self, file_name: str, printer: Printer, resin: Resin, 
                 layers: List[Image], intersection_levels: List[float],
                 layer_heights: List[float], volume: float):
        """
        Initialize file type with required parameters.
        
        Parameters
        ----------
        file_name: str
            Name of the file to be processed
        printer: Printer
            Printer object containing printer settings
        resin: Resin
            Resin object containing resin properties
        layers: List[Image]
            List of image layers
        intersection_levels: List[float]
            List of intersection levels
        layer_heights: List[float]
            List of layer heights
        volume: float
            Volume of the print
        """
        pass

    @abstractmethod 
    def save(self, save_path: str) -> bool:
        """
        Save file to the specified path.
        
        Parameters
        ----------
        save_path : str
            The path where the file should be saved.

        Returns
        -------
        bool
            True if the save was successful, False otherwise.
        """
        pass


class FileTypeRegistry:
    """
    A class to manage file type registrations.
    """
    _file_types: dict[str, type[FileType]] = {}

    @classmethod
    def register(cls, file_type: type[FileType]):
        cls._file_types[file_type.type_name] = file_type

    @classmethod
    def get(cls, type_name: str) -> type[FileType]:
        file_type = cls._file_types.get(type_name)
        if file_type is None:
            raise ValueError(f"File type '{type_name}' is not registered.")
        return file_type

    @classmethod
    def list_file_types(cls):
        return list(cls._file_types.values())