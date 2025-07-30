# OpenSLAice

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

<p align="center">
  <picture>
    <img src="https://www.cda.cit.tum.de/research/microfluidics/logo-microfluidics-toolkit.png" width="60%">
  </picture>
</p>

</div>

The MMFT OpenSLAice is a specialized slicer for masked sterolithography (mSLA) 3D printing of microfluidic devices. It addresses the unique challenges of printing microfluidic devices with high precision and reliability. It is part of the [Munich Microfluidics Toolkit (MMFT)](https://www.cda.cit.tum.de/research/microfluidics/munich-microfluidics-toolkit/) by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the Technical University of Munich.

## Overview

1. **Setup:**
   - *Initialization*: Printer (build volume, pixel resolution, z-stage movement) and resin (exposure time with regard to layer thickness) parameters are loaded.
   - *Geometry Import*: The 3D part (STL) is imported and metadata included.

2. **Core Slicing:**
   - *Feature Detection and Orientation*: The microfluidic channel network is detected and the 3D object is auto-oriented and placed accordingly to optimize channel alignment and reduce stair-step artifacts.
   - *Z-Slicing*: Layer heights for slicing are determined and the object is sliced along the z-axis into 2D planes, based on dynamic or static layer heights.
   - *XY-Rasterization*: Each 2D plane is converted into a bitmap mask aligned to the LCD pixel grid.

3. **Output and Calibration:**
   - *(Optional) Automatic Part Arrangement*: If several objects are printed at the same time, they are arranged on the build platform.
   - *Export*: The printer-specific export file including the printer, resin, and layer data is created and exported.
   - *(Optional) Exposure Calibration*: If necessary, the resin–printer combination can be calibrated using a single print evaluation.

## Contributors

OpenSLAice is a specialized slicer for microfluidic SLA/mSLA 3D printing, developed as part of a thesis project by Benjamin Liertz. 


## Getting Started

### Prerequisites

- Python 3.9+
- Jupyter Notebook

### Installation dependencies:

```bash
pip install -r requirements.txt
```

### Usage

#### Main Application

Open the main Jupyter notebook `OpenSLAice.ipynb` to access a simple example for how to use OpenSLAice.

#### Exposure Calibration

A specialized notebook is available for calibrating exposure `exp_calibration.ipynb`.

This notebook determine the needed resin model parameter used in the physical model for exposure time prediction. With these parameters and some metadata you can add any resin to the OpenSLAice database and use it for printing.

## Irradiance Measurements

The currently used resin parameters are where calibrated using the Anycubic Mono 4 Ultra with a measured irradiance of 1938.24 μW/cm² at 400nm at the FEP film. 
To ensure accurate exposure times across different printers, an irradiance sensor is provided to reuse the physical model.

In the `irradiance_sensor` directory, you'll find resources for building and using an irradiance sensor:

- **KiCad PCB Design**: Complete PCB design files for building a sensor board
- **Arduino Code**: Firmware for an ESP32-S3 SuperMinni with OPT3002 irradiance sensor
- **Usage Instructions**: How to measure and calibrate UV irradiance for your printer

This sensor allows precise measurement of your printer's light source intensity, enabling reuse of the physical model for exposure time prediction across different printers without needing to recalibrate.

## Project Structure

```
OpenSLAice/
├── examples/                # Example STL files and projects
│   ├── stls/                # Directory for STL files
│   │   ├── mixer.stl        # Example mixer STL
│   │   └── ELISA_chip.stl   # Example ELISA chip STL
│   └── ...                  # Other example projects
├── irradiance_sensor/       # Resources for irradiance sensor
│   ├── firmware/            # ESP32-S3 firmware for sensor
│   ├── kicad_pcb/           # KiCad PCB design files
│   └── README.md            # Sensor usage instructions
├── OpenSLAice.ipynb         # Main Jupyter notebook for OpenSLAice
├── exp_calibration.ipynb    # Jupyter notebook for exposure calibration
├── requirements.txt         # Python package dependencies
└── README.md                # Project overview and documentation
```

## Licensing

This repository is licensed under the MIT License. However, the following STL files are included under the Creative Commons Attribution 4.0 International License (CC BY 4.0):

- `examples/stls/mixer.stl`
- `examples/stls/ELISA_chip.stl`

See the [ATTRIBUTION.md](ATTRIBUTION.md) file for details.
