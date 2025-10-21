"""Simulates CT image acquisition using the GECATSim library.

This module provides the `Scanner` class, a high-level interface for configuring
and running virtual CT scans using GECATSim. It handles the entire pipeline,
from phantom preparation and voxelization to running the simulation,
reconstructing the image, and saving the output in DICOM format.

Key functionalities include:
- Reading and writing DICOM files.
- Converting NumPy arrays to DICOM series.
- Voxelizing phantoms for simulation based on material thresholds.
- Initializing and configuring a virtual CT scanner (`gecatsim.CatSim`).
- Running simulated scans in both axial and helical modes.
- Reconstructing CT images from raw projection data.
- Generating scout views for scan planning.
"""

from pathlib import Path
from shutil import rmtree
from datetime import datetime
from tempfile import TemporaryDirectory

import numpy as np
import matplotlib.pyplot as plt
import pydicom
import gecatsim as xc
from monai.data import MetaTensor

from gecatsim.reconstruction.pyfiles import recon
from .phantom import (voxelize_ground_truth,
                      Phantom)

install_path = Path(__file__).parent
available_scanners = [o.name for o in install_path.glob('defaults/*')
                      if not str(o).endswith('.cfg')]


def read_dicom(dcm_fname: str | Path) -> np.ndarray:
    """Reads a DICOM file and returns the pixel array in Hounsfield Units (HU).

    Args:
        dcm_fname (str | Path): The path to the DICOM file.

    Returns:
        np.ndarray: A NumPy array containing the pixel data, adjusted by the
            RescaleIntercept to be in HU.
    """
    dcm = pydicom.dcmread(str(dcm_fname))
    return dcm.pixel_array + int(dcm.RescaleIntercept)


def load_vol(file_list: list[str | Path]) -> np.ndarray:
    """Loads a list of DICOM files into a 3D NumPy array (volume).

    Args:
        file_list (list[str | Path]): A list of paths to the DICOM files
            that make up the volume.

    Returns:
        np.ndarray: A 3D NumPy array representing the stacked volume.
    """
    return np.stack(list(map(read_dicom, file_list)))


def convert_to_dicom(img_slice: np.ndarray, phantom_path: str | Path,
                     spacings: tuple[float, float, float]):
    """Converts a 2D NumPy array (image slice) into a DICOM file.

    A template DICOM file ("CT_small.dcm") is used and modified with the
    provided image data and metadata.

    Args:
        img_slice (np.ndarray): The input 2D NumPy array representing the
            image slice. Values are typically expected to be in HU.
        phantom_path (str | Path): The filename or path where the DICOM file
            will be saved.
        spacings (tuple[float, float, float]): A tuple containing voxel spacings
            in mm, ordered as (slice_thickness, pixel_spacing_rows,
            pixel_spacing_cols). Corresponds to (z, y, x).
    """
    # https://github.com/DIDSR/pediatricIQphantoms/blob/main/src/pediatricIQphantoms/make_phantoms.py#L144
    Path(phantom_path).parent.mkdir(exist_ok=True, parents=True)
    fpath = pydicom.data.get_testdata_file("CT_small.dcm")
    ds = pydicom.dcmread(fpath)
    ds.Rows, ds.Columns = img_slice.shape
    ds.SliceThickness = spacings[0]
    ds.PixelSpacing = [spacings[1], spacings[2]]
    # HU values must be adjusted by the RescaleIntercept before being stored.
    ds.PixelData = (img_slice.copy(order='C').astype('int16') -
                    int(ds.RescaleIntercept)).tobytes()
    pydicom.dcmwrite(phantom_path, ds)


def get_projection_data(ct: xc.CatSim) -> np.ndarray:
    """Reads raw projection data from the GECATSim simulation results.

    Args:
        ct (xc.CatSim): The GECATSim simulation object after a scan has been
            run. It is expected that `ct.resultsName` is set and the
            corresponding '.prep' file exists.

    Returns:
        np.ndarray: A NumPy array containing the projection data, typically
            with shape (viewCount, detectorRowCount, detectorColCount).

    Raises:
        FileNotFoundError: If the projection data file does not exist.
    """
    prep_file = ct.resultsName + '.prep'
    if not Path(prep_file).exists():
        raise FileNotFoundError(f"Projection data file not found: {prep_file}. "
                                "Ensure scan was run and resultsName is correct.")
    return xc.rawread(prep_file,
                      [ct.protocol.viewCount,
                       ct.scanner.detectorRowCount,
                       ct.scanner.detectorColCount],
                      'float')


def get_reconstructed_data(ct: xc.CatSim) -> np.ndarray:
    """Reads reconstructed image data from the GECATSim simulation results.

    Args:
        ct (xc.CatSim): The GECATSim simulation object after reconstruction.
            It's expected that `ct.resultsName` and reconstruction parameters
            (imageSize, sliceCount) are set, and the corresponding '.raw'
            reconstruction file exists.

    Returns:
        np.ndarray: A NumPy array containing the reconstructed image volume,
            typically with shape (sliceCount, imageSize, imageSize).

    Raises:
        FileNotFoundError: If the reconstructed data file does not exist.
    """
    imsize = ct.recon.imageSize
    recon_file = ct.resultsName + f'_{imsize}x{imsize}x{ct.recon.sliceCount}.raw'
    if not Path(recon_file).exists():
        raise FileNotFoundError(f"Reconstructed data file not found: {recon_file}. "
                                "Ensure recon was run and parameters are correct.")
    return xc.rawread(
        recon_file,
        [ct.recon.sliceCount, imsize, imsize], 'single')


def initialize_xcist(ground_truth_image: np.ndarray,
                     spacings: tuple[float, float, float] = (1.0, 1.0, 1.0),
                     scanner_model: str = 'Scanner_Default',
                     output_dir: str | Path = 'default_xcist_output',
                     phantom_id: str = 'default_phantom',
                     materials: dict | None = None) -> xc.CatSim:
    """Initializes a GECATSim (xc.CatSim) object for CT simulation.

    This function performs the initial setup for a simulation, including:
    1. Loading default configuration files.
    2. Setting up output directories.
    3. Converting the input `ground_truth_image` (in HU) into a series of
       DICOM slices.
    4. Voxelizing the DICOM series into a material-segmented phantom that
       GECATSim can use, based on the provided material thresholds.

    Args:
        ground_truth_image (np.ndarray): A 3D NumPy array representing the
            phantom in HU. Expected dimensions are (slices, rows, cols).
        spacings (tuple[float, float, float], optional): Voxel spacings in mm
            (z, y, x). Defaults to (1.0, 1.0, 1.0).
        scanner_model (str, optional): Name of the scanner configuration to
            load from the 'defaults' directory. Defaults to 'Scanner_Default'.
        output_dir (str | Path, optional): Base directory for simulation
            outputs. Defaults to 'default_xcist_output'.
        phantom_id (str, optional): Identifier for this phantom instance,
            used in filenames. Defaults to 'default_phantom'.
        materials (dict | None, optional): A dictionary mapping material names
            to their HU threshold values for segmentation. If None, default
            brain tissue thresholds are used by `voxelize_ground_truth`.

    Returns:
        xc.CatSim: The initialized GECATSim simulation object.
    """
    print('Initializing Scanner object...')
    print(''.join(10*['-']))
    # load defaults
    scanner_path = install_path / 'defaults' / scanner_model
    ct = xc.CatSim(install_path / 'defaults' / 'Phantom_Default',
                   install_path / 'defaults' / 'Physics_Default',
                   install_path / 'defaults' / 'Protocol_Default',
                   install_path / 'defaults' / 'Recon_Default',
                   scanner_path)

    ct.cfg.waitForKeypress = False
    ct.cfg.do_Recon = True

    # prepare directories
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    phantom_path = output_dir / 'phantoms' / f'{phantom_id}'
    phantom_path.mkdir(exist_ok=True, parents=True)
    ct.cfg.phantom.filename = str(phantom_path / f'{phantom_id}.json')

    # prepare material density arrays from ground truth phantom
    if ground_truth_image.ndim == 2:
        ground_truth_image = ground_truth_image[None]

    dicom_path = phantom_path / 'dicom'
    for slice_id, img in enumerate(ground_truth_image):
        dicom_filename = dicom_path / f'1-{slice_id:03d}.dcm'
        convert_to_dicom(img, dicom_filename, spacings=spacings)
    voxelize_ground_truth(dicom_path, phantom_path,
                          material_threshold_dict=materials)
    print('Scanner Ready')
    return ct


class Scanner():
    """Manages CT simulation, from setup to reconstruction and DICOM output.

    This class acts as a high-level wrapper around the GECATSim library,
    simplifying the process of running virtual imaging trials. It takes a
    `Phantom` object and handles the GECATSim configuration, scan execution,
    image reconstruction, and final output.

    Attributes:
        phantom (Phantom): The phantom object to be scanned.
        scanner_model (str): Name of the GECATSim scanner configuration.
        output_dir (Path): Root directory for all simulation outputs.
        xcist (xc.CatSim): The underlying GECATSim simulation object.
        recon (np.ndarray | None): Stores the reconstructed image volume.
        projections (np.ndarray | None): Stores the raw projection data.
        start_positions (np.ndarray): Calculated Z-axis start positions for
            a series of axial scans.
        total_scan_length (float): Total length of the phantom along the Z-axis.
        pitch (float): The pitch value for helical scans (0 for axial).
        kVp (float): The kVp value used for the most recent scan.
        tempdir (TemporaryDirectory | None): Manages a temporary directory for
            outputs if `output_dir` is not specified.
    """
    kernels = ['standard', 'soft', 'bone', 'R-L', 'S-L']

    def __init__(self, phantom: Phantom, scanner_model: str = "Scanner_Default",
                 studyname: str = "default_study",
                 studyid: int = 0, seriesname: str = "default_series", seriesid: int = 0,
                 framework: str = 'CATSIM', output_dir: str | Path | None = None,
                 materials: dict | None = None) -> None:
        """Initializes the Scanner object.

        Args:
            phantom (Phantom): An instance of the `Phantom` class containing
                the image data (in HU) and metadata.
            scanner_model (str, optional): Name of the GECATSim scanner config
                (e.g., 'Scanner_Default') or a path to a config directory.
                Defaults to "Scanner_Default".
            studyname (str, optional): Study name for DICOM metadata.
                Defaults to "default_study".
            studyid (int, optional): Study ID for DICOM metadata. Defaults to 0.
            seriesname (str, optional): Series name for DICOM metadata.
                Defaults to "default_series".
            seriesid (int, optional): Series ID for DICOM metadata.
                Defaults to 0.
            framework (str, optional): The simulation framework to use.
                Currently, only 'CATSIM' is supported. Defaults to 'CATSIM'.
            output_dir (str | Path | None, optional): Directory to save all
                simulation results. If None, a temporary directory is created.
                Defaults to None.
            materials (dict | None, optional): Dictionary for material
                segmentation during phantom voxelization. Passed to
                `initialize_xcist`. Defaults to None.

        Raises:
            FileNotFoundError: If the specified `scanner_model` is not found.
            ValueError: If an unsupported `framework` is specified.
        """
        if output_dir is None:
            self.tempdir = TemporaryDirectory()
            output_dir = self.tempdir.name
        else:
            self.tempdir = None # type: ignore
        output_dir = Path(output_dir) / f'{phantom.patient_name}'
        if output_dir.exists():
            rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir = output_dir

        img = phantom.get_CT_number_phantom()
        if isinstance(img, MetaTensor):
            img = img.numpy()

        self.phantom = phantom
        self.scanner_model = scanner_model if scanner_model in available_scanners else 'Scanner_Default'
        if (scanner_model not in available_scanners) and (not Path(scanner_model).exists()):
            raise FileNotFoundError(f'{scanner_model} not in {available_scanners} and is not a file')
        self.studyname = studyname or phantom.patient_name
        self.studyid = studyid
        self.seriesname = seriesname
        self.seriesid = seriesid
        if framework != 'CATSIM':
            raise ValueError("Only 'CATSIM' framework is currently supported.")
        self.framework = framework
        self.recon = None
        self.projections = None
        self.total_scan_length = self.phantom.spacings[0] * self.phantom.shape[0]
        self.pitch = 1.0
        self.xcist = initialize_xcist(img, self.phantom.spacings,
                                      scanner_model=self.scanner_model,
                                      output_dir=self.output_dir,
                                      phantom_id=str(phantom.patientid),
                                      materials=materials)
        if Path(scanner_model).exists():
            self.load_scanner_config(scanner_model)
        self.start_positions = self.calculate_start_positions()

    @property
    def nominal_aperature(self) -> float:
        """Calculates the nominal collimated beam width (aperture) at isocenter.

        This is derived from the detector row size, detector row count,
        and the system magnification (SDD/SID).

        Returns:
            float: The nominal aperture in mm.
        """
        M = self.xcist.cfg.scanner.sdd / self.xcist.cfg.scanner.sid
        slice_thickness = self.xcist.cfg.scanner.detectorRowSize / M
        return slice_thickness * self.xcist.cfg.scanner.detectorRowCount

    def load_scanner_config(self, filename: str | Path = 'Scanner_Default'):
        """Loads a specific scanner hardware configuration into GECATSim.

        Args:
            filename (str | Path): The name of a scanner config directory in
                `defaults` (e.g., 'GE_Lightspeed16') or a direct path to one.

        Returns:
            Scanner: The Scanner instance for method chaining.

        Raises:
            FileNotFoundError: If the specified config is not found.
        """
        if filename in available_scanners:
            filename = install_path / 'defaults' / filename
        cfg = xc.source_cfg(filename)
        self.xcist.cfg.scanner = cfg.scanner
        self.scanner_model = Path(filename).name
        return self

    def calculate_start_positions(self, startZ: float | None = None,
                                  endZ: float | None = None) -> np.ndarray:
        """Determines Z-axis start positions for a series of axial scans.

        This calculates the table positions needed to cover a specified range
        of the phantom, bounded by the phantom's total Z-length.

        Args:
            startZ (float | None, optional): Desired starting Z position (mm)
                relative to phantom center (0). Defaults to the phantom's
                negative z-extent.
            endZ (float | None, optional): Desired ending Z position (mm).
                Defaults to the phantom's positive z-extent.

        Returns:
            np.ndarray: An array of Z positions (mm) where each axial scan
                should start.
        """
        if startZ is None:
            startZ = -self.total_scan_length / 2
        else:
            startZ = max(-self.total_scan_length / 2, startZ)
        if endZ is None:
            endZ = self.total_scan_length / 2
        else:
            endZ = min(endZ, self.total_scan_length / 2)
        return np.arange(startZ, endZ, self.nominal_aperature)

    def recommend_scan_range(self, threshold: float = -950) -> tuple[int, int]:
        """Recommends a scan range based on a scout view attenuation profile.

        It analyzes the mean attenuation along the phantom's Z-axis to find the
        extent of material above a given HU threshold, suggesting a tighter
        scan range.

        Args:
            threshold (float, optional): The HU threshold to distinguish
                material from air. Defaults to -950.

        Returns:
            tuple[int, int]: Recommended (startZ, endZ) in mm, as integers.
        """
        img = np.array(self.phantom.get_CT_number_phantom())
        scout_profile = img.mean(axis=(1, 2))
        active_voxels = scout_profile > threshold
        if not np.any(active_voxels):
            return (int(self.start_positions[0]), int(self.start_positions[-1]))

        suggested_start_idx = np.argmax(active_voxels)
        suggested_start_mm = self.start_positions[0] + suggested_start_idx * self.phantom.spacings[0]

        # Find the last index where the profile is above the threshold
        suggested_end_idx = len(active_voxels) - 1 - np.argmax(active_voxels[::-1])
        suggested_end_mm = self.start_positions[0] + suggested_end_idx * self.phantom.spacings[0]

        return (int(suggested_start_mm), int(suggested_end_mm))

    def scout_view(self, startZ: float | None = None, endZ: float | None = None,
                   pitch: float = 0.0):
        """Displays a simulated scout radiograph of the phantom.

        This shows a projection image (summed along one axis) overlaid with
        indicators for the current scan range and other scan parameters.

        Args:
            startZ (float | None, optional): Starting Z position for the scan
                range overlay. Defaults to the calculated start.
            endZ (float | None, optional): Ending Z position for the scan
                range overlay. Defaults to the calculated end.
            pitch (float, optional): Pitch value to display table speed for
                helical scan planning. Defaults to 0.0 (axial).
        """
        start_positions = self.calculate_start_positions(startZ, endZ)
        img = self.phantom.get_CT_number_phantom()
        plt.imshow(img.sum(axis=1), cmap='gray', origin='lower',
                   extent=[-img.shape[2] * self.phantom.spacings[2] / 2,
                           img.shape[2] * self.phantom.spacings[2] / 2,
                           self.start_positions[0] + self.total_scan_length,
                           self.start_positions[0]])
        plt.hlines(y=max(start_positions[0], -self.total_scan_length/2),
                   xmin=-img.shape[2]*self.phantom.spacings[2] / 2,
                   xmax=img.shape[2]*self.phantom.spacings[2]/2, color='red')
        plt.annotate('Stop', (0, start_positions[0]-10),
                     horizontalalignment='center')

        plt.hlines(y=min(start_positions[-1] + self.nominal_aperature,
                         self.total_scan_length/2),
                   xmin=-img.shape[2]*self.phantom.spacings[2]/2,
                   xmax=img.shape[2]*self.phantom.spacings[2]/2, color='red')
        plt.annotate('Start', (0, start_positions[-1] + self.nominal_aperature + 10),
                     horizontalalignment='center')
        rotations = len(self.calculate_start_positions(startZ, endZ))
        if pitch:
            rotations /= pitch
        plt.annotate(f'{rotations:.2f} scans required',
                     xy=(0, (start_positions[0]+start_positions[-1])/2),
                     horizontalalignment='center')
        plt.annotate('', xy=(40, start_positions[-1] + self.nominal_aperature),
                     xytext=(40, start_positions[0]),
                     arrowprops=dict(facecolor='black', shrink=0.05))
        speed = pitch*self.nominal_aperature/self.xcist.cfg.protocol.rotationTime
        plt.title(f'Table Speed: {speed:.2f} mm/s')
        plt.ylabel('scan z position [mm]')
        plt.xlabel('scan x position [mm]')

    def __repr__(self) -> str:
        """Returns a string representation of the Scanner object."""
        repr_str = f'''
        {self.__class__.__name__} {self.seriesname}
        Scanner: {self.scanner_model}
        Simulation Platform: {self.framework}
        '''
        if self.recon is None:
            return repr_str
        repr_str += f'\nRecon: {self.recon.shape} {self.xcist.cfg.recon.fov/10} cm fov'
        if self.projections is None:
            return repr_str
        repr_str += f'\nProjections: {self.projections.shape}'
        return repr_str

    def run_scan(self, mA: float = 200, kVp: int = 120,
                 startZ: float | None = None, endZ: float | None = None,
                 views: int | None = None, pitch: float = 0, bhc: bool | str = True):
        """Runs the CT simulation with the specified parameters.

        Args:
            mA (float, optional): X-ray source milliamps. Higher mA reduces
                noise. Defaults to 200.
            kVp (int, optional): X-ray source potential. Affects beam energy
                and contrast. Must be one of [70, 80, ..., 140]. Defaults to 120.
            startZ (float | None, optional): Starting table position (mm) for
                the scan. Defaults to the first calculated position.
            endZ (float | None, optional): Ending table position (mm) for the
                scan. Defaults to the last calculated position.
            views (int | None, optional): Number of views per rotation. Reducing
                can speed up tests but may cause aliasing. Defaults to None,
                using the protocol's default.
            pitch (float, optional): The ratio of table travel per rotation to
                beam width. A pitch of 0 indicates an axial scan. A pitch > 0
                indicates a helical scan. Defaults to 0.
            bhc (bool | str, optional): Beam hardening correction.
                - `True`: Applies a polynomial correction.
                - `False`: Disables BHC.
                - `'default'`: Uses XCIST's default BHC (can cause capping).
                Defaults to True.

        Returns:
            Scanner: The Scanner instance for method chaining.

        Raises:
            ValueError: If an unsupported `kVp` or `pitch` is selected.
        """
        self.xcist.cfg.protocol.mA = mA
        kVp_options = [70, 80, 90, 100, 110, 120, 130, 140]
        kVp = int(kVp)
        if kVp not in kVp_options:
            raise ValueError(f'Selected kVP [{kVp}] not available, please choose from {kVp_options}')
        spectrum_file = f'tungsten_tar7.0_{int(kVp)}_filt.dat'
        spectrum_path = Path(xc.__file__).parent / 'spectrum' / spectrum_file
        if not spectrum_path.exists():
            raise FileNotFoundError(f"Spectrum file not found: {spectrum_path}")
        self.xcist.cfg.protocol.spectrumFilename = str(spectrum_path)
        self.kVp = kVp

        if views:
            self.xcist.cfg.protocol.viewsPerRotation = views

        if pitch < 0:
            raise ValueError(f'pitch: {pitch} must be >= 0')
        self.pitch = pitch

        if bhc is True:
            self.xcist.cfg.physics.callback_post_log = 'Prep_BHC_Accurate'
            self.xcist.cfg.physics.EffectiveMu = 0.2
            self.xcist.cfg.physics.BHC_poly_order = 5
            self.xcist.cfg.physics.BHC_max_length_mm = int(self.phantom.size[1])
            self.xcist.cfg.physics.BHC_length_step_mm = 10
        elif bhc is False:
            self.xcist.cfg.physics.callback_post_log = ""
            self.xcist.cfg.protocol.bowtie = ""
        # 'default' will just use the default setting in the cfg.

        self.results_dir = self.output_dir / 'simulations' / f'{self.phantom.patientid}'
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.xcist.cfg.experimentDirectory = str(self.results_dir)
        proj_file = str((self.results_dir / f'{mA}mA_{kVp}kV').absolute())
        self.xcist.cfg.resultsName = proj_file
        self.xcist.resultsName = proj_file
        startZ = self.start_positions[0] if startZ is None else startZ
        endZ = self.start_positions[-1] if endZ is None else endZ
        self.scan_coverage = (startZ, endZ)
        if pitch == 0:  # axial case
            self._projections = self.axial_scan(startZ, endZ)
        else:  # helical
            self._projections = self.helical_scan(startZ, endZ, pitch)
        return self

    def axial_scan(self, startZ: float, endZ: float) -> list[str]:
        """Runs an axial scan acquisition.

        The scanner performs a series of acquisitions at discrete table
        positions to cover the specified Z-range.

        Args:
            startZ (float): The starting Z-position (mm) of the scan.
            endZ (float): The ending Z-position (mm) of the scan.

        Returns:
            list[str]: A list of paths to the generated projection files.
        """
        self.xcist.cfg.protocol.tableSpeed = 0
        self.xcist.cfg.protocol.viewCount = self.xcist.cfg.protocol.viewsPerRotation
        self.xcist.cfg.protocol.startViewId = 0
        self.xcist.protocol.stopViewId = self.xcist.cfg.protocol.startViewId + self.xcist.cfg.protocol.viewCount - 1

        start_positions = self.calculate_start_positions(startZ, endZ)
        projections = []
        for idx, table_position in enumerate(start_positions):
            print(f'scan: {idx+1}/{len(start_positions)}')
            proj_file = str((self.results_dir /
                            f'{idx:03d}_{Path(self.xcist.resultsName).name}'
                             ).absolute())
            self.xcist.cfg.resultsName = proj_file
            self.xcist.resultsName = proj_file
            self.xcist.protocol.startZ = table_position
            self.xcist.run_all()
            projections.append(proj_file)
        return projections

    def helical_scan(self, startZ: float, endZ: float, pitch: float) -> list[str]:
        """Runs a helical scan acquisition.

        The scanner table moves continuously while the gantry rotates to
        acquire data over the specified Z-range.

        Args:
            startZ (float): The starting Z-position (mm) of the scan.
            endZ (float): The ending Z-position (mm) of the scan.
            pitch (float): The pitch value for the helical scan.

        Returns:
            list[str]: A list containing the path to the projection file.
        """
        self.xcist.cfg.protocol.scanTrajectory = "Gantry_Helical"
        self.xcist.cfg.protocol.startZ = startZ
        self.startZ, self.endZ = startZ, endZ
        self.xcist.cfg.protocol.tableSpeed = pitch * self.nominal_aperature / self.xcist.cfg.protocol.rotationTime
        table_travel_per_rotation = self.xcist.cfg.protocol.tableSpeed * self.xcist.cfg.protocol.rotationTime
        scan_length = endZ - startZ
        rotations = np.ceil(scan_length / table_travel_per_rotation).astype(int)
        if rotations % 2 == 0:
            rotations += 1

        self.xcist.cfg.protocol.viewCount = self.xcist.cfg.protocol.viewsPerRotation * rotations
        self.xcist.cfg.protocol.startViewId = 0
        self.xcist.cfg.protocol.stopViewId = self.xcist.cfg.protocol.startViewId + self.xcist.cfg.protocol.viewCount - 1
        self.xcist.run_all()
        return [self.xcist.cfg.resultsName]

    def run_recon(self, fov: float | None = None, slice_thickness: float | None = None,
                  slice_increment: float | None = None, mu_water: float | None = None,
                  kernel: str = 'standard'):
        """Performs image reconstruction and stores it in `self.recon`.

        Args:
            fov (float | None, optional): Field of View (mm) for reconstruction.
                Defaults to None, using the config default.
            slice_thickness (float | None, optional): Nominal width (mm) of the
                reconstructed slice along the z-axis. Defaults to `slice_increment`.
            slice_increment (float | None, optional): Distance (mm) between
                consecutive reconstructed slices. Defaults to `slice_thickness`.
            mu_water (float | None, optional): The linear attenuation
                coefficient (mu) of water for HU scaling. If None, it's
                calculated based on the monochromatic energy. Defaults to None.
            kernel (str, optional): Reconstruction kernel. Options include:
                'standard', 'soft', 'bone', 'R-L' (Ram-Lak), 'S-L' (Shepp-Logan).
                Defaults to 'standard'.

        Returns:
            Scanner: The Scanner instance for method chaining.

        Raises:
            ValueError: If an unsupported `kernel` is specified.
        """
        if kernel not in self.kernels:
            raise ValueError(f'{kernel} not in {self.kernels}')
        self.xcist.cfg.recon.kernelType = kernel

        slice_thickness = slice_thickness or self.xcist.cfg.recon.sliceThickness
        slice_increment = slice_increment or slice_thickness

        if slice_increment:
            # Note: XCIST's recon.sliceThickness is used for slice increment.
            self.xcist.recon.sliceThickness = slice_increment

        if mu_water:
            self.xcist.cfg.recon.mu = mu_water
        elif self.xcist.physics.monochromatic != -1:
            self.xcist.recon.mu = xc.GetMu('water', self.xcist.physics.monochromatic)[0] / 10

        if fov:
            self.xcist.cfg.recon.fov = fov
        print(f'fov size: {self.xcist.cfg.recon.fov}')
        self.xcist.cfg.recon.displayImagePictures = False

        if self.pitch > 0:
            recon_volume = self.helical_recon()
        else:
            recon_volume = self.axial_recon()

        # Average slices together to form thicker slices
        if slice_increment and slice_thickness:
            recons = []
            n_slices = len(recon_volume)
            starts = np.arange(0, n_slices, int(slice_increment))
            for slab_start in starts:
                slab_end = slab_start + int(slice_thickness)
                if slab_end <= n_slices:
                    recons.append(recon_volume[slab_start:slab_end].mean(axis=0))
            self.recon = np.stack(recons) if recons else np.array([])
        else:
            self.recon = recon_volume

        self.projections = get_projection_data(self.xcist)
        self.groundtruth = None  # Clear previous ground truth if any
        return self

    def axial_recon(self) -> np.ndarray:
        """Performs reconstruction for an axial scan.

        Returns:
            np.ndarray: The reconstructed 3D image volume.
        """
        self.xcist.cfg.recon.reconType = 'fdk_equiAngle'
        scan_width = 0.8 * self.nominal_aperature
        valid_slices = int(scan_width // self.xcist.recon.sliceThickness)
        self.xcist.cfg.recon.sliceCount = valid_slices
        recons = []
        for proj in self._projections:
            self.xcist.cfg.resultsName = proj
            self.xcist.resultsName = self.xcist.cfg.resultsName
            vol = recon.recon_direct(self.xcist.cfg).transpose(2, 0, 1)
            recons.append(vol)
        return np.concatenate(recons, axis=0) if recons else np.array([])

    def helical_recon(self) -> np.ndarray:
        """Performs reconstruction for a helical scan.

        Returns:
            np.ndarray: The reconstructed 3D image volume.

        Raises:
            ValueError: If called for a scan with pitch=0.
        """
        if not self.pitch:
            raise ValueError('Helical recon requires scan data with pitch>0.')
        self.xcist.cfg.recon.reconType = 'helical_equiAngle'
        exam_range = self.endZ - self.startZ
        sliceCount = int(exam_range / self.xcist.cfg.recon.sliceThickness)
        self.xcist.cfg.recon.sliceCount = sliceCount
        return recon.recon_direct(self.xcist.cfg).transpose(2, 0, 1)

    def write_to_dicom(self, fname: str | Path,
                       groundtruth: bool = False) -> list[Path]:
        """Writes the reconstructed or ground truth CT data to a DICOM series.

        Args:
            fname (str | Path): The base filename for the DICOM series. Slices
                will be numbered (e.g., `base_000.dcm`, `base_001.dcm`).
            groundtruth (bool, optional): If True, saves the ground truth
                phantom image instead of the reconstructed image. Defaults to False.

        Returns:
            list[Path]: A list of paths to the written DICOM files.

        Adapted from:
        https://github.com/DIDSR/pediatricIQphantoms/blob/main/src/pediatricIQphantoms/make_phantoms.py#L144
        """
        fpath = pydicom.data.get_testdata_file("CT_small.dcm")
        ds = pydicom.dcmread(fpath)
        # Update metadata
        ds.Manufacturer = 'GE (simulated)'
        ds.ManufacturerModelName = 'LightSpeed 16 (simulated)'
        time = datetime.now()
        ds.InstanceCreationDate = time.strftime('%Y%m%d')
        ds.InstanceCreationTime = time.strftime('%H%M%S')
        ds.InstitutionName = 'FDA/CDRH/OSEL/DIDSR'
        ds.StudyDate = ds.InstanceCreationDate
        ds.StudyTime = ds.InstanceCreationTime
        ds.PatientName = self.phantom.patient_name
        ds.SeriesNumber = self.seriesid
        ds.PatientAge = f'{int(self.phantom.age):03d}Y'
        ds.PatientID = f'{int(self.phantom.patientid):03d}'
        if 'PatientWeight' in ds: del ds.PatientWeight
        if 'ContrastBolusRoute' in ds: del ds.ContrastBolusRoute
        if 'ContrastBolusAgent' in ds: del ds.ContrastBolusAgent
        ds.ImageComments = f"effctive diameter [cm]: {self.phantom.size[1]/10}"
        ds.ScanOptions = self.xcist.cfg.protocol.scanTrajectory.upper()
        ds.ReconstructionDiameter = self.xcist.cfg.recon.fov
        ds.ConvolutionKernel = self.xcist.cfg.recon.kernelType
        ds.Exposure = self.xcist.cfg.protocol.mA

        vol = self.groundtruth if groundtruth else self.recon
        if vol is None:
            raise ValueError("No volume available to save. Run reconstruction first.")
        if vol.ndim == 2:
            vol = vol[None]
        nslices, ds.Rows, ds.Columns = vol.shape

        ds.SpacingBetweenSlices = self.xcist.cfg.recon.sliceThickness
        ds.DistanceSourceToDetector = self.xcist.cfg.scanner.sdd
        ds.DistanceSourceToPatient = self.xcist.cfg.scanner.sid
        ds.PixelSpacing = [self.xcist.cfg.recon.fov / self.xcist.cfg.recon.imageSize,
                           self.xcist.cfg.recon.fov / self.xcist.cfg.recon.imageSize]
        ds.SliceThickness = self.xcist.cfg.recon.sliceThickness
        ds.KVP = self.kVp
        ds.StudyID = str(self.studyid)

        # Create unique UIDs for series and study
        base_uid_part = '.'.join(ds.SeriesInstanceUID.split('.')[:-1])
        ds.SeriesInstanceUID = f"{base_uid_part}.{self.studyid}{self.seriesid}"
        base_uid_part = '.'.join(ds.StudyInstanceUID.split('.')[:-1])
        ds.StudyInstanceUID = f"{base_uid_part}.{self.studyid}"
        ds.AcquisitionNumber = self.studyid

        fname = Path(fname)
        fname.parent.mkdir(exist_ok=True, parents=True)
        fnames = []
        for slice_idx, array_slice in enumerate(vol):
            ds.InstanceNumber = slice_idx + 1
            # Create unique SOP Instance UID for each slice
            sop_base_uid = '.'.join(ds.SOPInstanceUID.split('.')[:-1])
            ds.SOPInstanceUID = f"{sop_base_uid}.{self.studyid}{self.seriesid}{slice_idx}"
            ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID

            # Set slice location and position
            start_z = self.scan_coverage[0] if hasattr(self, 'scan_coverage') else -self.total_scan_length / 2
            slice_location = start_z + (slice_idx * ds.SliceThickness)
            ds.SliceLocation = slice_location
            ds.ImagePositionPatient = [-ds.Rows/2*ds.PixelSpacing[0],
                                       -ds.Columns/2*ds.PixelSpacing[1],
                                       slice_location]
            ds.PixelData = (array_slice.copy(order='C').astype('int16') -
                            int(ds.RescaleIntercept)).tobytes()

            dcm_fname = fname.parent / f'{fname.stem}_{slice_idx:03d}{fname.suffix}' if nslices > 1 else fname
            fnames.append(dcm_fname)
            pydicom.dcmwrite(dcm_fname, ds)
        return fnames