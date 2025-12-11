"""Converts a series of DICOM images into a voxelized phantom for XCIST.

This script reads a directory of DICOM images and a set of parameters from a
configuration object. It then segments the images into different materials based
on Hounsfield Unit (HU) thresholds and generates a set of output files that
represent a voxelized phantom, suitable for use in the XCIST CT simulation
framework.

The primary entry point is the `DICOM_to_voxelized_phantom` function.

Inputs (specified in a configuration object, e.g., `cfg.phantom`):
    dicom_path (str): Path to the directory containing the DICOM images.
    phantom_path (str): Path where the output phantom files will be written.
    materials (list[str]): List of material names (e.g., 'ncat_water').
    mu_energy (float): The energy (keV) at which to calculate material
        attenuation coefficients (mu), corresponding to the effective energy
        of the source DICOM scan.
    thresholds (list[float]): Lower HU thresholds for each material.
    slice_range (list[int]): Range of DICOM image numbers to include.
    show_phantom (bool): If True, display the resulting phantom slices.
    overwrite (bool): If True, overwrite existing output files without prompting.

Outputs:
    - A .json file describing the phantom's properties.
    - A .raw file for the input HU images.
    - A .raw file for each material, containing the volume fraction of that
      material in each voxel.
"""

from pathlib import Path
import os
from argparse import ArgumentParser
import numpy as np
import re
import pydicom
import copy
import json
import matplotlib.pyplot as plt
from gecatsim import GetMu, source_cfg


class IndexTracker:
    """A class to track and update the displayed slice in a matplotlib plot.

    This class enables scrolling through slices of a 3D image volume using
    the mouse wheel in a matplotlib window.

    Attributes:
        ax (matplotlib.axes.Axes): The matplotlib axes object for the plot.
        X (np.ndarray): The 3D image data, expected shape (slices, rows, cols).
        slices (int): The number of slices in the image data.
        ind (int): The index of the currently displayed slice.
        im (matplotlib.image.AxesImage): The image object being displayed.
    """
    def __init__(self, ax, X):
        """Initializes the IndexTracker.

        Args:
            ax (matplotlib.axes.Axes): The axes on which to display the image.
            X (np.ndarray): The 3D image volume to scroll through.
        """
        self.ax = ax
        self.X = X
        self.slices, _, _ = X.shape
        self.ind = self.slices // 2
        self.im = ax.imshow(self.X[self.ind, :, :])
        self.update()

    def on_scroll(self, event):
        """Handles the mouse scroll event to change the displayed slice.

        Args:
            event (matplotlib.backend_bases.MouseEvent): The scroll event.
        """
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        """Updates the displayed image to the current slice index."""
        self.im.set_data(self.X[self.ind, :, :])
        self.ax.set_ylabel(f'slice {self.ind}')
        self.im.axes.figure.canvas.draw()


def initialize(phantom) -> tuple:
    """Initializes variables and prepares for phantom generation.

    This function performs several setup tasks:
    - Validates the DICOM path and creates a sorted list of DICOM files.
    - Reads a sample DICOM to extract metadata (dimensions, spacing, etc.).
    - Sets up output paths and filenames.
    - Calculates material attenuation coefficients (mu) and segmentation
      thresholds if not provided.
    - Creates the output directory and handles overwriting existing files.

    Args:
        phantom: A configuration object with attributes like `dicom_path`,
            `phantom_path`, `materials`, `mu_energy`, etc.

    Returns:
        A tuple containing:
        - The updated phantom configuration object.
        - A list of DICOM filenames to be processed.
        - A dictionary to hold the volume fraction arrays for each material.
        - A dictionary containing material properties (names, mu values, etc.).
    """
    if not os.path.exists(phantom.dicom_path):
        raise FileNotFoundError(f"DICOM path not found: {phantom.dicom_path}")

    all_files = [f for f in os.listdir(phantom.dicom_path) if os.path.isfile(os.path.join(phantom.dicom_path, f))]
    dcm_files = [j for j in all_files if j.endswith('.dcm')]
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in {phantom.dicom_path}")

    # Sort DICOM files numerically instead of alphabetically
    dcm_files = sorted(dcm_files)

    if hasattr(phantom, 'slice_range'):
        indices = list(range(phantom.slice_range[0], phantom.slice_range[1] + 1))
        dcm_files = [dcm_files[i] for i in indices]

    sample_dicom = pydicom.dcmread(os.path.join(phantom.dicom_path, dcm_files[0]))

    # Initialize phantom properties from DICOM metadata
    phantom.basename = os.path.basename(phantom.phantom_path)
    phantom.num_materials = len(phantom.materials)
    phantom.num_cols, phantom.num_rows = sample_dicom.Columns, sample_dicom.Rows
    phantom.num_slices = len(dcm_files)
    phantom.pixel_size_x, phantom.pixel_size_y = sample_dicom.PixelSpacing
    phantom.pixel_size_z = sample_dicom.SliceThickness
    phantom.mu_water = GetMu('water', phantom.mu_energy)[-1]
    phantom.json_filename = os.path.join(phantom.phantom_path, f"{phantom.basename}.json")

    # Prepare filenames and data structures
    base_fname = os.path.join(phantom.phantom_path, f"{phantom.basename}_")
    suffix = f"_{phantom.num_cols}x{phantom.num_rows}x{phantom.num_slices}.raw"
    mu_list = [GetMu(mat, phantom.mu_energy)[0] for mat in phantom.materials]

    volume_fraction_array = {mat: np.zeros((phantom.num_slices, phantom.num_rows, phantom.num_cols), dtype=np.float32) for mat in phantom.materials}
    volume_fraction_array['HU data'] = np.zeros((phantom.num_slices, phantom.num_rows, phantom.num_cols), dtype=np.float32)

    volume_fraction_filenames = [f"{base_fname}{mat}{suffix}" for mat in phantom.materials]
    volume_fraction_filenames.append(f"{base_fname}HU_data{suffix}")

    # Sort materials by mu value to define thresholds between them
    sorted_indices = sorted(range(len(mu_list)), key=lambda k: mu_list[k])
    sorted_materials = [phantom.materials[i] for i in sorted_indices]
    sorted_mu = [mu_list[i] for i in sorted_indices]

    # Calculate thresholds based on midpoints between sorted mu values
    calculated_thresholds = [0]
    for i in range(1, len(sorted_mu)):
        calculated_thresholds.append(sorted_mu[i-1] * 0.55 + sorted_mu[i] * 0.45)

    materials_dict = {
        'material_names': sorted_materials,
        'volume_fraction_filenames': volume_fraction_filenames,
        'mu_values': sorted_mu,
        'threshold_values': calculated_thresholds
    }

    # Override calculated thresholds if provided in config
    if hasattr(phantom, 'thresholds') and len(phantom.thresholds) == len(materials_dict['threshold_values']):
        mu_thresholds = [(hu + 1000) * phantom.mu_water / 1000 for hu in phantom.thresholds]
        materials_dict['threshold_values'] = mu_thresholds

    # Handle output directory creation and overwrite logic
    os.makedirs(phantom.phantom_path, exist_ok=True)
    if not getattr(phantom, 'overwrite', False) and any(os.path.exists(f) for f in volume_fraction_filenames):
        print("Warning: Output files exist and will be overwritten.")
        input("Press Enter to continue or Ctrl-C to quit.")

    return phantom, dcm_files, volume_fraction_array, materials_dict


def compute_volume_fraction_array(phantom, dicom_filenames, materials_dict, volume_fraction_array):
    """Computes the volume fraction for each material from DICOM slices.

    This function iterates through each DICOM file, converts the HU pixel data
    to linear attenuation coefficients (mu), and then segments the image into
    material fractions based on the provided thresholds.

    Args:
        phantom: The phantom configuration object.
        dicom_filenames (list[str]): List of DICOM filenames to process.
        materials_dict (dict): Dictionary containing material properties.
        volume_fraction_array (dict): Dictionary of NumPy arrays to store the
            output volume fractions.

    Returns:
        dict: The updated `volume_fraction_array` dictionary filled with
              computed data.
    """
    thresholds = materials_dict['threshold_values']
    material_names = materials_dict['material_names']
    mu_values = materials_dict['mu_values']
    print(f"* Calculating volume fraction maps for {len(thresholds)} materials and {len(dicom_filenames)} slices...")

    for i, filename in enumerate(dicom_filenames):
        dicom_path = os.path.join(phantom.dicom_path, filename)
        dcm_data = pydicom.dcmread(dicom_path)

        hu_array = dcm_data.pixel_array.astype(np.float32) + int(dcm_data.RescaleIntercept)
        volume_fraction_array['HU data'][i] = hu_array

        mu_array = (hu_array + 1000) * phantom.mu_water / 1000
        mu_array[mu_array < 0] = 0

        bounds = copy.deepcopy(thresholds) + [1.1 * mu_array.max()]

        for j, material in enumerate(material_names):
            # Isolate pixels within the material's mu range
            mask = (mu_array >= bounds[j]) & (mu_array < bounds[j+1])
            material_slice = np.zeros_like(mu_array)
            material_slice[mask] = mu_array[mask] / mu_values[j]
            volume_fraction_array[material][i] = material_slice

    return volume_fraction_array


def write_files(phantom, materials_dict, volume_fraction_array):
    """Writes the volume fraction arrays and the JSON metadata file.

    Args:
        phantom: The phantom configuration object.
        materials_dict (dict): Dictionary of material properties.
        volume_fraction_array (dict): Dictionary containing the computed
            volume fraction data.
    """
    filenames = materials_dict['volume_fraction_filenames']
    # The HU data is the last item in the array and filenames list
    all_materials = materials_dict['material_names'] + ['HU data']
    print(f"* Writing volume fraction files for {phantom.num_materials} materials and {phantom.num_slices} slices, plus HU data...")

    for i, material_name in enumerate(all_materials):
        filename = filenames[i]
        print(f"* Writing {filename}...")
        with open(filename, 'wb') as f:
            f.write(volume_fraction_array[material_name].tobytes())
   
    write_json_file(phantom, materials_dict)


def write_json_file(phantom, materials_dict):
    """Generates and writes the phantom's JSON descriptor file.

    Args:
        phantom: The phantom configuration object.
        materials_dict (dict): Dictionary of material properties.
    """
    # Use only the material filenames, not the HU data filename
    material_filenames = [Path(f).name for f in materials_dict['volume_fraction_filenames'][:-1]]

    json_contents = {
        "construction_description": "Created by dicom_to_voxelized_phantom.py",
        "n_materials": phantom.num_materials,
        "mat_name": materials_dict['material_names'],
        "mu_values": materials_dict['mu_values'],
        "mu_thresholds": materials_dict['threshold_values'],
        "volumefractionmap_filename": material_filenames,
        "volumefractionmap_datatype": ["float"] * phantom.num_materials,
        "cols": [phantom.num_cols] * phantom.num_materials,
        "rows": [phantom.num_rows] * phantom.num_materials,
        "slices": [phantom.num_slices] * phantom.num_materials,
        "x_size": [phantom.pixel_size_x] * phantom.num_materials,
        "y_size": [phantom.pixel_size_y] * phantom.num_materials,
        "z_size": [phantom.pixel_size_z] * phantom.num_materials,
        "x_offset": [0.5 + phantom.num_cols / 2] * phantom.num_materials,
        "y_offset": [0.5 + phantom.num_rows / 2] * phantom.num_materials,
        "z_offset": [0.5 + phantom.num_slices / 2] * phantom.num_materials,
    }

    print(f"* Writing {phantom.json_filename}...")
    with open(phantom.json_filename, 'w') as f:
        json.dump(json_contents, f, indent=4)


def DICOM_to_voxelized_phantom(phantom):
    """Main function to orchestrate the DICOM to voxelized phantom conversion.

    Args:
        phantom: A configuration object containing all necessary parameters.
    """
    phantom, dicom_filenames, volume_fraction_array, materials_dict = initialize(phantom)

    volume_fraction_array = compute_volume_fraction_array(phantom, dicom_filenames, materials_dict, volume_fraction_array)
    
    write_files(phantom, materials_dict, volume_fraction_array)

    if getattr(phantom, 'show_phantom', False):
        num_materials = phantom.num_materials
        if num_materials <= 3:
            rows, cols = 1, num_materials
        elif num_materials == 4:
            rows, cols = 2, 2
        else: # Default for > 4
            rows, cols = 2, (num_materials + 1) // 2

        fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, squeeze=False)
        axes = axes.flatten()
        trackers = []
        for i, material in enumerate(materials_dict['material_names']):
            ax = axes[i]
            tracker = IndexTracker(ax, volume_fraction_array[material])
            trackers.append(tracker)
            fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
            ax.set_title(material)
        plt.show()


def run_from_config(config_filename: str):
    """Loads a configuration file and runs the phantom conversion.

    Args:
        config_filename (str): Path to the configuration file.
    """
    cfg = source_cfg(config_filename)
    DICOM_to_voxelized_phantom(cfg.phantom)


if __name__ == "__main__":
    parser = ArgumentParser(
        prog='Dicom to Voxelized Phantom for XCIST',
        description='Converts DICOM series to an XCIST voxelized phantom based on a config file.'
    )
    parser.add_argument('filename', help='Path to the configuration file.')
    args = parser.parse_args()
    run_from_config(args.filename)