"""VITools: A Python package for Virtual Imaging Trials.

This package provides a high-level API for setting up and running CT imaging
simulations using the XCIST framework. It simplifies the creation of phantoms,
configuration of scanners, and management of large-scale simulation studies.

The main components exposed to the user are:
- `Phantom`: A class to represent the object being imaged.
- `Scanner`: A class to control the CT simulation process.
- `Study`: A class to manage and execute a series of simulations.
- Helper functions like `read_dicom`, `load_vol`, and `get_available_phantoms`.
"""
from .phantom import Phantom
from .scanner import Scanner, read_dicom, available_scanners, load_vol
from .study import Study, get_available_phantoms