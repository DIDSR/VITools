'''
module for working with phantoms
'''

from pathlib import Path

import numpy as np
from monai.transforms import Resize
import gecatsim as xc

from . import dicom_to_voxelized_phantom


def resize(phantom, shape, **kwargs):
    resize = Resize(max(shape), size_mode='longest', **kwargs)
    resized = resize(phantom[None])[0]
    return resized


def voxelize_ground_truth(dicom_path: str | Path, phantom_path: str | Path,
                          material_threshold_dict: dict | None = None):
    '''
    Used to convert ground truth image into segmented volumes used by XCIST to
    run simulations

    :param dicom_path: str | Path, path where the DICOM images are located,
        these are typically the output of `convert_to_dicom`
    :param phantom_path: str or Path, where the phantom files are to be
        written
    :param material_threshold_dict: dictionary mapping XCIST materials to
        appropriate lower thresholds in the ground truth image, see the .cfg
        here for examples
        <https://github.com/xcist/phantoms-voxelized/tree/main/DICOM_to_voxelized>
    '''
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
    cfg.phantom.dicom_path = dicom_path
    cfg.phantom.phantom_path = phantom_path
    cfg.phantom.materials = list(material_threshold_dict.keys())
    cfg.phantom.mu_energy = 60
    cfg.phantom.thresholds = list(material_threshold_dict.values())
    cfg.phantom.slice_range = [slice_range[0], slice_range[-1]]
    cfg.phantom.show_phantom = False
    cfg.phantom.overwrite = True
    dicom_to_voxelized_phantom.DICOM_to_voxelized_phantom(cfg.phantom)


class Phantom:
    '''
    A base phantom that accepts any image array and spacings to define its size.

    :param img: numpy.ndarray, 2D or 3D, defining the phantom
    :param spacings: tuple, voxel spacings [mm] (z, x, y).
                    Default is 1 mm in each direction.
    :param patient_name: str, patient identifier to be saved in DICOM header.
                        Default is 'default'.
    :param patientid: int, patient identifier to be saved in DICOM header.
                     Default is 0.
    :param age: float, patient age in years to be saved in DICOM header.
                Default is 0.
    '''
    def __init__(self, img: np.ndarray, spacings: tuple = (1, 1, 1),
                 patient_name: str = 'default', patientid: int = 0, age: float = 0) -> None:
        self._phantom = img
        self.dz, self.dx, self.dy = spacings
        self.nz, self.nx, self.ny = self._phantom.shape
        self.patient_name = patient_name
        self.patientid = patientid
        self.age = age

    def __repr__(self) -> str:
        string_representation = f'''
        Phantom Class: {self.__class__.__name__}
        Age (years): {self.age}
        Shape (voxels): {self.shape}
        Size (mm): {self.size}
        '''
        return string_representation

    def get_CT_number_phantom(self) -> np.ndarray:
        '''Returns the phantom array'''
        return self._phantom

    @property
    def spacings(self) -> tuple:
        '''Returns the voxel spacings (z, x, y)'''
        return self.dz, self.dx, self.dy

    @property
    def shape(self) -> tuple:
        '''Returns the shape of the phantom array'''
        return self.get_CT_number_phantom().shape

    @property
    def size(self) -> np.ndarray:
        '''Returns the size of the phantom array (mm)'''
        return tuple(map(lambda o: float(round(o, ndigits=2)),
                         np.array(self.spacings) * list(self.shape)))

    def resize(self, shape: tuple, **kwargs) -> None:
        '''
        Resizes the phantom array to the given shape and adjusts the spacings accordingly.

        :param shape: tuple, new shape for the phantom array
        '''
        original_shape = np.array(self.shape)
        self._phantom = resize(self._phantom, shape, **kwargs)
        new_shape = np.array(self._phantom.shape)
        new_spacings = original_shape / new_shape * np.array(self.spacings)
        self.dz, self.dx, self.dy = new_spacings
        self.nz, self.nx, self.ny = shape
