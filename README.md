<div align="center">

# OpenSLAice

**A specialized slicer for microfluidic SLA/mSLA 3D printing**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

</div>

## Overview

OpenSLAice is a specialized slicer for microfluidic SLA/mSLA 3D printing, developed as part of a bachelor thesis project. It addresses the unique challenges of printing microfluidic devices with high precision and reliability.

### Key Innovations

1. **Physical Model-Based Calibration**: Single-print calibration method that predicts exposure times for arbitrary layer heights based on a physical model of light penetration.

2. **Automated Microfluidic Feature Detection**: Automatically detects and orients microfluidic features to minimize stair-stepping artifacts, resulting in smoother channel walls.

3. **Dynamic Layer Height Selection**: Intelligently balances print quality with speed by varying layer heights throughout the print - using fine layers for precise channels and thicker layers for stable channel top layers.

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
├── requirements.txt          # Python package dependencies
└── README.md                # Project overview and documentation
```

## Licensing

This repository is licensed under the MIT License. However, the following STL files are included under the Creative Commons Attribution 4.0 International License (CC BY 4.0):

- `examples/stls/mixer.stl`
- `examples/stls/ELISA_chip.stl`

See the [ATTRIBUTION.md](ATTRIBUTION.md) file for details.
