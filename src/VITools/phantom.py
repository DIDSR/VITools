"""Defines the Phantom class and related utility functions.

This module provides the core components for representing and manipulating
phantoms in virtual imaging trials. It includes the `Phantom` class, which
encapsulates the phantom's image data and metadata, and helper functions for
resizing and converting DICOM images to voxelized phantoms compatible with
the XCIST simulation framework.
"""

from pathlib import Path

import numpy as np
from monai.transforms import Resize
import gecatsim as xc

from . import dicom_to_voxelized_phantom


def resize(phantom: np.ndarray, shape: tuple, **kwargs) -> np.ndarray:
    """Resizes a phantom to a new shape while maintaining aspect ratio.

    This function uses MONAI's Resize transform to resize a 2D or 3D phantom
    array. The `size_mode='longest'` option scales the longest dimension to
    match the corresponding dimension in `shape`, and scales other dimensions
    proportionally.

    mode = 'nearest' is useful for downsizing without interpolation errors

    Args:
        phantom (np.ndarray): The phantom image array to resize.
        shape (tuple): The target shape for the phantom.
        **kwargs: Additional keyword arguments to be passed to
            `monai.transforms.Resize`. 
            E.g.: `from monai.transforms import Resize; Resize?`

    Returns:
        np.ndarray: The resized phantom array.
    """
    resize_transform = Resize(max(shape), size_mode='longest', **kwargs)
    # MONAI transforms expect a channel dimension, so we add and remove one.
    resized = resize_transform(phantom[None])[0]
    return resized


def voxelize_ground_truth(dicom_path: str | Path, phantom_path: str | Path,
                          material_threshold_dict: dict | None = None):
    """Converts a ground truth DICOM series into a voxelized phantom for XCIST.

    This function takes a series of DICOM images and segments them into
    different materials based on Hounsfield Unit (HU) thresholds. The
    resulting voxelized phantom can be used in XCIST simulations.

    Args:
        dicom_path (str | Path): Path to the directory containing the DICOM
            series. These are typically the output of `convert_to_dicom`.
        phantom_path (str | Path): Path to the directory where the output
            phantom files will be written.
        material_threshold_dict (dict | None, optional): A dictionary mapping
            material names (e.g., 'ncat_adipose') to their lower HU threshold
            values. If None, a default dictionary for brain tissue is used.
            For examples, see:
            https://github.com/xcist/phantoms-voxelized/tree/main/DICOM_to_voxelized
    """
    nfiles = len(list(Path(dicom_path).rglob('*.dcm')))
    slice_range = list(range(nfiles))
    if not material_threshold_dict:
        material_threshold_dict = dict(zip(
                                        ['ncat_adipose',
                                         'ncat_water',
                                         'ncat_brain',
                                         'ncat_skull'],
                                        [-200, -10, 10, 300]))
    cfg = xc.CFG()
    cfg.phantom.dicom_path = str(dicom_path)
    cfg.phantom.phantom_path = str(phantom_path)
    cfg.phantom.materials = list(material_threshold_dict.keys())
    cfg.phantom.mu_energy = 60
    cfg.phantom.thresholds = list(material_threshold_dict.values())
    cfg.phantom.slice_range = [slice_range[0], slice_range[-1]]
    cfg.phantom.show_phantom = False
    cfg.phantom.overwrite = True
    dicom_to_voxelized_phantom.DICOM_to_voxelized_phantom(cfg.phantom)


class Phantom:
    """A base class for representing a medical phantom.

    This class encapsulates a 2D or 3D image array, its voxel spacings, and
    patient-related metadata.

    Attributes:
        dz (float): Voxel spacing in the z-direction (mm).
        dx (float): Voxel spacing in the x-direction (mm).
        dy (float): Voxel spacing in the y-direction (mm).
        nz (int): Number of voxels in the z-direction.
        nx (int): Number of voxels in the x-direction.
        ny (int): Number of voxels in the y-direction.
        patient_name (str): Patient identifier for DICOM headers.
        patientid (int): Patient ID for DICOM headers.
        age (float): Patient age in years for DICOM headers.
    """
    def __init__(self, img: np.ndarray, spacings: tuple = (1, 1, 1),
                 patient_name: str = 'default', patientid: int = 0, age: float = 0) -> None:
        """Initializes the Phantom object.

        Args:
            img (np.ndarray): A 2D or 3D NumPy array representing the phantom's
                CT numbers.
            spacings (tuple, optional): A tuple of voxel spacings in (z, x, y)
                order (mm). Defaults to (1, 1, 1).
            patient_name (str, optional): Patient identifier to be saved in the
                DICOM header. Defaults to 'default'.
            patientid (int, optional): Patient identifier to be saved in the
                DICOM header. Defaults to 0.
            age (float, optional): Patient age in years to be saved in the
                DICOM header. Defaults to 0.
        """
        self._phantom = img
        self.dz, self.dx, self.dy = spacings
        self.nz, self.nx, self.ny = self._phantom.shape
        self.patient_name = patient_name
        self.patientid = patientid
        self.age = age

    def __repr__(self) -> str:
        """Returns a string representation of the Phantom object."""
        string_representation = f'''
        Phantom Class: {self.__class__.__name__}
        Age (years): {self.age}
        Shape (voxels): {self.shape}
        Size (mm): {self.size}
        '''
        return string_representation

    def get_CT_number_phantom(self) -> np.ndarray:
        """Returns the phantom's CT number array."""
        return self._phantom

    @property
    def spacings(self) -> tuple:
        """Returns the voxel spacings (z, x, y) in mm."""
        return self.dz, self.dx, self.dy

    @property
    def shape(self) -> tuple:
        """Returns the shape of the phantom array in (nz, nx, ny) order."""
        return self.get_CT_number_phantom().shape

    @property
    def size(self) -> np.ndarray:
        """Returns the physical size of the phantom (mm) as a tuple (z, x, y)."""
        return tuple(map(lambda o: float(round(o, ndigits=2)),
                         np.array(self.spacings) * list(self.shape)))

    def resize(self, shape: tuple, **kwargs) -> None:
        """Resizes the phantom and adjusts voxel spacings.

        This method resizes the internal phantom array to the given shape and
        recalculates the voxel spacings to maintain the phantom's physical size.

        Args:
            shape (tuple): The new target shape for the phantom array
                (nz, nx, ny).
            **kwargs: Additional keyword arguments passed to the `resize`
                function, which in turn passes them to
                `monai.transforms.Resize`.
        """
        original_shape = np.array(self.shape)
        self._phantom = resize(self._phantom, shape, **kwargs)
        new_shape = np.array(self._phantom.shape)
        new_spacings = original_shape / new_shape * np.array(self.spacings)
        self.dz, self.dx, self.dy = new_spacings
        self.nz, self.nx, self.ny = self._phantom.shape