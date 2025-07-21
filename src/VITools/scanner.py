"""
This module is responsible for simulating CT image acquisition using the GECATSim library.

It provides functionalities for:
- Reading and writing DICOM files.
- Preparing and voxelizing phantoms for simulation.
- Initializing and configuring a virtual CT scanner (`gecatsim.CatSim`).
- Running simulated scans (axial and helical).
- Reconstructing CT images from projection data.
- Generating scout views.
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
    '''
    Reads a DICOM file and returns the pixel array adjusted by the RescaleIntercept.

    Args:
        dcm_fname (str | Path): The filename or path to the DICOM file to be read.

    Returns:
        np.ndarray: A NumPy array containing the pixel data in Hounsfield Units (HU).
    '''
    dcm = pydicom.dcmread(str(dcm_fname))
    return dcm.pixel_array + int(dcm.RescaleIntercept)


def load_vol(file_list):
    return np.stack(list(map(read_dicom, file_list)))


def convert_to_dicom(img_slice: np.ndarray, phantom_path: str | Path,
                     spacings: tuple[float, float, float]):
    '''
    Converts a 2D NumPy array (image slice) into a DICOM file.

    A template DICOM file ("CT_small.dcm") is used and modified with the
    provided image data and metadata.

    Args:
        img_slice (np.ndarray): The input 2D NumPy array representing the image slice.
                                Values are typically expected to be in HU.
        phantom_path (str | Path): The filename or path where the DICOM file will be saved.
        spacings (tuple[float, float, float]): A tuple containing pixel spacings in mm,
                                               ordered as (slice_thickness, pixel_spacing_rows,
                                               pixel_spacing_cols).
                                               Example: (z_spacing, y_spacing, x_spacing)
    '''
    # https://github.com/DIDSR/pediatricIQphantoms/blob/main/src/pediatricIQphantoms/make_phantoms.py#L144
    Path(phantom_path).parent.mkdir(exist_ok=True, parents=True)
    fpath = pydicom.data.get_testdata_file("CT_small.dcm")
    ds = pydicom.dcmread(fpath)
    ds.Rows, ds.Columns = img_slice.shape
    ds.SliceThickness = spacings[0]
    ds.PixelSpacing = [spacings[1], spacings[2]]
    ds.PixelData = (img_slice.copy(order='C').astype('int16') -\
        int(ds.RescaleIntercept)).tobytes()
    pydicom.dcmwrite(phantom_path, ds)


def get_projection_data(ct: xc.CatSim) -> np.ndarray:
    '''
    Reads raw projection data from the simulation results of a GECATSim object.

    Args:
        ct (xc.CatSim): The GECATSim simulation object after a scan has been run.
                        It is expected that `ct.resultsName` is set and the
                        corresponding '.prep' file exists.

    Returns:
        np.ndarray: A NumPy array containing the projection data.
                    The shape is typically (viewCount, detectorRowCount, detectorColCount).
    '''
    prep_file = ct.resultsName + '.prep'
    if not Path(prep_file).exists():
        raise FileNotFoundError(f"Projection data file not found: {prep_file}. Ensure scan was run and resultsName is correct.")
    return xc.rawread(prep_file,
                      [ct.protocol.viewCount,
                       ct.scanner.detectorRowCount,
                       ct.scanner.detectorColCount],
                      'float')


def get_reconstructed_data(ct: xc.CatSim) -> np.ndarray:
    '''
    Reads reconstructed image data from the simulation results of a GECATSim object.

    Args:
        ct (xc.CatSim): The GECATSim simulation object after reconstruction.
                        It's expected that `ct.resultsName` and reconstruction parameters
                        (imageSize, sliceCount) are set, and the corresponding '.raw'
                        reconstruction file exists.

    Returns:
        np.ndarray: A NumPy array containing the reconstructed image volume.
                    The shape is typically (sliceCount, imageSize, imageSize).
    '''
    imsize = ct.recon.imageSize
    recon_file = ct.resultsName + f'_{imsize}x{imsize}x{ct.recon.sliceCount}.raw'
    if not Path(recon_file).exists():
        raise FileNotFoundError(f"Reconstructed data file not found: {recon_file}. Ensure recon was run and parameters are correct.")

    return xc.rawread(
        recon_file,
        [ct.recon.sliceCount, imsize, imsize], 'single')


def initialize_xcist(ground_truth_image: np.ndarray,
                     spacings: tuple[float, float, float] = (1.0, 1.0, 1.0),
                     scanner_model: str = 'Scanner_Default',
                     output_dir: str | Path = 'default_xcist_output',
                     phantom_id: str = 'default_phantom',
                     materials: dict | None = None) -> xc.CatSim:
    '''
    Initializes a GECATSim (xc.CatSim) object for CT simulation.

    This involves:
    1. Loading default configuration files for phantom, physics, protocol, recon, and scanner.
    2. Setting up output directories.
    3. Converting the input `ground_truth_image` (HU values) into DICOM slices.
    4. Voxelizing these DICOM slices into a material-segmented phantom compatible with GECATSim,
       using the provided `materials` dictionary for segmentation thresholds.

    Args:
        ground_truth_image (np.ndarray): A 3D NumPy array representing the phantom in HU.
                                         Expected dimensions are (slices, rows, cols) or (z, y, x).
        spacings (tuple[float, float, float], optional):
            Voxel spacings in mm (slice_thickness, pixel_spacing_rows, pixel_spacing_cols).
            Corresponds to (z_spacing, y_spacing, x_spacing). Defaults to (1.0, 1.0, 1.0).
        scanner_model (str, optional): Name of the scanner configuration to load from
                                       the 'defaults' directory. Defaults to 'Scanner_Default'.
        output_dir (str | Path, optional): Base directory for simulation outputs.
                                           Defaults to 'default_xcist_output'.
        phantom_id (str, optional): Identifier for this phantom instance, used in filenames.
                                    Defaults to 'default_phantom'.
        materials (dict | None, optional):
            A dictionary mapping material names (str) to their HU threshold values (int/float).
            Used by `voxelize_ground_truth` to segment the HU image into material fractions.
            Example: `{'air': -800, 'soft_tissue': -100, 'bone': 200}`.
            If None, default thresholds in `voxelize_ground_truth` might be used.

    Returns:
        xc.CatSim: The initialized GECATSim simulation object.
    '''
    print('Initializing Scanner object...')
    print(''.join(10*['-']))
    # load defaults
    scanner_path = install_path / 'defaults'/ scanner_model

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
    dicom_path = fr'{dicom_path}'
    voxelize_ground_truth(dicom_path, phantom_path,
                          material_threshold_dict=materials)
    print('Scanner Ready')
    return ct


class Scanner():
    """
    A class to encapsulate CT simulation parameters, manage the GECATSim object,
    and run various simulation tasks like scanning, reconstruction, and DICOM output.

    Attributes:
        phantom (Phantom): The phantom object to be scanned.
        scanner_model (str): Name of the GECATSim scanner configuration.
        output_dir (Path): Directory for all outputs related to this scanner instance.
        xcist (xc.CatSim): The underlying GECATSim simulation object.
        recon (np.ndarray | None): Stores the reconstructed image volume.
        projections (np.ndarray | None): Stores the raw projection data.
        start_positions (np.ndarray): Calculated start Z positions for axial scans.
        total_scan_length (float): Total length of the phantom along the Z-axis.
        pitch (float): Current pitch value for helical scans.
        kVp (float): Current kVp value used for the scan.
        tempdir (TemporaryDirectory | None): Handles temporary directory if output_dir is not specified.
    """

    kernels = ['standard', 'soft', 'bone', 'R-L', 'S-L']

    def __init__(self, phantom: Phantom, scanner_model: str = "Scanner_Default",
                 studyname: str = "default_study",
                 studyid: int = 0, seriesname: str = "default_series", seriesid: int = 0,
                 framework: str = 'CATSIM', output_dir: str | Path | None = None,
                 materials: dict | None = None) -> None:
        """
        Initializes the Scanner object.

        Args:
            phantom (Phantom): An instance of the `Phantom` class containing the image data
                               (in HU) and metadata (spacings, patient info).
            scanner_model (str, optional):
                Name of the GECATSim scanner configuration to use (e.g., 'Scanner_Default').
                These configurations are expected to be in `install_path / 'defaults'`.
                Alternatively, a direct path to a scanner configuration directory can be provided.
                Defaults to "Scanner_Default".
            studyname (str, optional): Study identifier for DICOM metadata. Defaults to "default_study".
            studyid (int, optional): Study ID for DICOM metadata. Defaults to 0.
            seriesname (str, optional): Series identifier for DICOM metadata. Defaults to "default_series".
            seriesid (int, optional): Series ID for DICOM metadata. Defaults to 0.
            framework (str, optional): CT simulation framework. Currently only 'CATSIM' (GECATSim)
                                       is supported. Defaults to 'CATSIM'.
            output_dir (str | Path | None, optional):
                Directory to save intermediate and final simulation results.
                If None, a temporary directory will be created and used.
                A subdirectory named after `phantom.patient_name` will be created within this.
                Defaults to None.
            materials (dict | None, optional):
                Dictionary for material segmentation during phantom voxelization by `initialize_xcist`.
                Example: `{'air': -800, 'soft_tissue': -100, 'bone': 200}`.
                Defaults to None (which might use defaults in `voxelize_ground_truth`).

        Raises:
            FileNotFoundError: If the specified `scanner_model` (as a name or path) is not found.
            ValueError: If an unsupported `framework` is specified.
        """
        if output_dir is None:
            self.tempdir = TemporaryDirectory()
            output_dir = self.tempdir.name
        output_dir = Path(output_dir) / f'{phantom.patient_name}'
        if output_dir.exists():
            rmtree(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir = output_dir

        img = phantom.get_CT_number_phantom()
        if isinstance(img, MetaTensor):
            img = img.numpy()

        self.phantom = phantom
        self.scanner_model = scanner_model if scanner_model in available_scanners else 'Scanner_Default'
        if (scanner_model not in available_scanners) and (not Path(scanner_model).exists()):
            raise FileNotFoundError(f'{scanner_model} not in {available_scanners} and is not a file')
        self.studyname = studyname or self.patientname
        self.studyid = studyid
        self.seriesname = seriesname
        self.seriesid = seriesid
        self.framework = framework
        self.ndetectors = 900
        self.nangles = 1000
        self.detector_size = 1
        self.recon = None
        self.projections = None
        self.groundtruth = None
        self.patient_diameter = 18
        self.zspan = 'dynamic'
        # self.table_speed = 'Intermediate'
        self.total_scan_length = self.phantom.spacings[0]*self.phantom.shape[0]
        self.pitch = 1
        self.xcist = initialize_xcist(img, self.phantom.spacings,
                                      scanner_model=self.scanner_model,
                                      output_dir=self.output_dir,
                                      phantom_id=phantom.patientid,
                                      materials=materials)
        if Path(scanner_model).exists():
            self.load_scanner_config(scanner_model)
        self.start_positions = self.calculate_start_positions()

    @property
    def nominal_aperature(self) -> float:
        """
        Calculates the nominal collimated beam width (aperture) at the isocenter.

        This is derived from the detector row size, detector row count,
        and the system magnification (SDD/SID).

        Returns:
            float: The nominal aperture in mm.
        """
        M = self.xcist.cfg.scanner.sdd/self.xcist.cfg.scanner.sid
        slice_thickness = self.xcist.cfg.scanner.detectorRowSize/M
        return slice_thickness*self.xcist.cfg.scanner.detectorRowCount

    def load_scanner_config(self, filename: str = 'Scanner_Default'):
        """
        Loads a specific scanner hardware configuration into the GECATSim object.

        Args:
            filename (str | Path):
                The name of a scanner configuration directory within `install_path / 'defaults'`
                (e.g., 'Scanner_Default'), or a direct path to a scanner configuration directory.
                This directory should contain the `Scanner_*.cfg` file.

        Returns:
            Scanner: The Scanner instance itself, allowing for method chaining.

        Raises:
            FileNotFoundError: If the specified scanner configuration is not found.
        """
        if filename in available_scanners:
            filename = install_path / 'defaults' / filename
        cfg = xc.source_cfg(filename)
        self.xcist.cfg.scanner = cfg.scanner
        self.scanner_model = Path(filename).name
        return self

    def calculate_start_positions(self, startZ=None, endZ=None):
        '''
        Determines the Z-axis start positions for a series of axial scans
        required to cover a specified portion of the phantom.

        The scan range is bounded by the phantom's total Z-length.

        Args:
            startZ (float | None, optional):
                Desired starting Z position (in mm) of the scan coverage, relative to phantom center (0).
                If None, defaults to the negative half of the total scan length.
            endZ (float | None, optional):
                Desired ending Z position (in mm) of the scan coverage, relative to phantom center (0).
                If None, defaults to the positive half of the total scan length.

        Returns:
            np.ndarray: An array of Z positions (in mm) where each axial scan acquisition should start.
        '''
        if startZ is None:
            startZ = -self.total_scan_length/2
        else:
            startZ = max(-self.total_scan_length/2, startZ)
        if endZ is None:
            endZ = self.total_scan_length/2
        else:
            endZ = min(endZ, self.total_scan_length/2)
        return np.arange(startZ, endZ, self.nominal_aperature)

    def recommend_scan_range(self, threshold=-950) -> tuple:
        '''
        Recommends a scan range (startZ, endZ) based on a scout view attenuation profile.

        It analyzes the mean attenuation along the Z-axis of the phantom's CT number image
        to find the extent of attenuating material above a given HU threshold.

        Args:
            threshold (float, optional):
                The Hounsfield Unit (HU) threshold used to distinguish attenuating material
                (e.g., tissue) from air-like regions. Defaults to -950 HU.

        Returns:
            tuple[float, float]: Recommended (startZ, endZ) in mm.
        '''
        img = self.phantom.get_CT_number_phantom()
        scout_profile = img.mean(axis=(1, 2))
        suggested_start_idx = np.argmax(np.diff(scout_profile > threshold)) + 1
        suggested_start_mm = self.start_positions[0] + suggested_start_idx *\
            self.phantom.spacings[0]

        if np.all(scout_profile[suggested_start_idx:] > threshold):
            suggested_end_mm = self.start_positions[-1]
        else:
            suggested_end_idx = \
                np.argmax(np.diff(scout_profile[suggested_start_idx:] >
                                  threshold)) + 1
            suggested_end_mm = self.start_positions[0] + suggested_end_idx *\
                self.phantom.spacings[0]
        return (int(suggested_start_mm), int(suggested_end_mm))

    def scout_view(self, startZ=None, endZ=None, pitch=0):
        '''
        Displays a simulated scout radiograph (summation along one axis) of the phantom,
        overlaid with indicators for the current scan range (startZ, endZ) and
        the number of axial scans or table speed for helical scans.

        Args:
            startZ (float | None, optional):
                Starting Z position for the scan range overlay.
                If None, uses the first position from `self.calculate_start_positions()`.
            endZ (float | None, optional):
                Ending Z position for the scan range overlay.
                If None, uses the last position from `self.calculate_start_positions()`.
            pitch (float, optional):
                Pitch value to calculate table speed for helical scan indication.
                Defaults to 0.0 (axial).
        '''
        start_positions = self.calculate_start_positions(startZ, endZ)
        img = self.phantom.get_CT_number_phantom()
        plt.imshow(img.sum(axis=1), cmap='gray', origin='lower',
                   extent=[-img.shape[0]*self.phantom.spacings[0]/2,
                           img.shape[0]*self.phantom.spacings[0]/2,
                           self.start_positions[0]+self.total_scan_length,
                           self.start_positions[0]])
        plt.hlines(y=max(start_positions[0], -self.total_scan_length/2),
                   xmin=-img.shape[0]*self.phantom.spacings[0] / 2,
                   xmax=img.shape[0]*self.phantom.spacings[0]/2, color='red')
        plt.annotate('Stop', (0, start_positions[0]-10),
                     horizontalalignment='center')

        plt.hlines(y=min(start_positions[-1] + self.nominal_aperature,
                         self.total_scan_length/2),
                   xmin=-img.shape[0]*self.phantom.spacings[0]/2,
                   xmax=img.shape[0]*self.phantom.spacings[0]/2, color='red')
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
        repr = f'''
        {self.__class__} {self.seriesname}
        Scanner: {self.scanner_model}
        Simulation Platform: {self.framework}
        '''
        if self.recon is None:
            return repr
        repr += f'\nRecon: {self.recon.shape} {self.xcist.cfg.recon.fov/10} cm fov'
        if self.projections is None:
            return repr
        repr += f'\nProjections: {self.projections.shape}'
        return repr

    def run_scan(self, mA=200, kVp=120, startZ=None, endZ=None, views=None,
                 pitch=0, bhc=True):
        """
        Runs the CT simulation using the stored parameters.

        :param mA: x-ray source milliamps, increases x-ray flux linearly,
            $noise ~ 1/sqrt(mA)$
        :param kVp: x-ray source potential, increases x-ray flux
            nonlinearly and reduces contrast as increased
        :param startZ: optional starting table position in mm of the scan,
            see self.start_positions
        :param endZ: optional last position of scan in mm,
            see self.start_positions
        :param views: number of angular views per rotation, for testing this can be
            reduced but will produced aliasing streaks
        :param verbose: optional boolean, if True prints out status
            updates, if False they are suppressed.
        :param pitch: optional [float] table travel during helical CT;
            equal to table travel (mm) per gantry rotation / total nominal beam width (mm)
            pitch = 0 for axial scan mode, pitch > 0 for helical scans.
        :param bhc: optional [bool, str], if True applies polynomial beam
            hardening correction to correct for polychromatic cupping artifact,
            if `default` applied default XCIST bhc, caution this gives capping
            artifacts. Options include [True, False, 'default']
        """
        self.xcist.cfg.protocol.mA = mA
        kVp_options = [70, 80, 90, 100, 110, 120, 130, 140]
        if kVp not in kVp_options:
            raise ValueError(f'Selected kVP [{kVp}] not available,\
                              please choose from {kVp_options}')
        self.xcist.cfg.protocol.spectrumFilename =\
            f'tungsten_tar7.0_{kVp}_filt.dat'
        self.kVp = kVp

        if views:
            self.xcist.cfg.protocol.viewsPerRotation = views

        if pitch >= 0:
            self.pitch = pitch
        else:
            raise ValueError(f'pitch: {pitch} must be >=0')

        if bhc is True:
            self.xcist.cfg.physics.callback_post_log = 'Prep_BHC_Accurate'
            self.xcist.cfg.physics.EffectiveMu = 0.2
            self.xcist.cfg.physics.BHC_poly_order = 5
            self.xcist.cfg.physics.BHC_max_length_mm = int(self.phantom.size[1])
            self.xcist.cfg.physics.BHC_length_step_mm = 10
        elif bhc is False:
            self.xcist.cfg.physics.callback_post_log = ""
            self.xcist.cfg.protocol.bowtie = ""

        self.results_dir = self.output_dir / 'simulations' / \
            f'{self.phantom.patientid}'
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.xcist.protocol.spectrumFilename = f"tungsten_tar7.0_{int(kVp)}_filt.dat"  # name of the spectrum file
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

    def axial_scan(self, startZ, endZ):
        '''runs axial scan mode

        runs axial scan from `startZ` [mm] to endZ [mm]
        '''
        self.xcist.cfg.protocol.tableSpeed = 0
        self.xcist.cfg.protocol.viewCount =\
            self.xcist.cfg.protocol.viewsPerRotation
        self.xcist.cfg.protocol.startViewId = 0
        self.xcist.protocol.stopViewId =\
            self.xcist.cfg.protocol.startViewId + self.xcist.cfg.protocol.viewCount - 1

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

    def helical_scan(self, startZ, endZ, pitch):
        '''runs helical scan mode

        runs helical scan from `startZ` [mm] to endZ [mm]
        '''
        self.xcist.cfg.protocol.scanTrajectory = "Gantry_Helical"
        self.xcist.cfg.protocol.startZ = startZ

        self.startZ, self.endZ = startZ, endZ
        self.xcist.cfg.protocol.tableSpeed = pitch*self.nominal_aperature/self.xcist.cfg.protocol.rotationTime # v = h*s/t
        table_travel_per_rotation = self.xcist.cfg.protocol.tableSpeed*self.xcist.cfg.protocol.rotationTime # table_travel_per_rotation == d == v*t
        scan_length = endZ - startZ
        rotations = np.ceil(scan_length / table_travel_per_rotation).astype(int)
        if rotations % 2 == 0:
            rotations += 1

        self.xcist.cfg.protocol.viewCount = self.xcist.cfg.protocol.viewsPerRotation*rotations
        self.xcist.cfg.protocol.startViewId = 0
        self.xcist.cfg.protocol.stopViewId = self.xcist.cfg.protocol.startViewId + self.xcist.cfg.protocol.viewCount - 1
        self.xcist.run_all()
        projections = [self.xcist.cfg.resultsName]
        return projections

    def run_recon(self, fov=None, slice_thickness=None, slice_increment=None,
                  mu_water=None, kernel='standard'):
        '''
        perform reconstruction and save to .recon attribute

        :param kernel: reconstruction kernel, options include: ['standard', 'soft', 'bone', 'R-L', 'S-L']
            'R-L' for Ramachandran-Lakshminarayanan (R-L) filter
            'S-L' for Shepp-Logan (S-L) filter
            See https://github.com/xcist/main/blob/master/gecatsim/cfg/Recon_Default.cfg
        :param slice_thickness: float, thickness in mm of slice nominal width of reconstructed image along the z axis
        :param slice_increment: float, Distance [mm] between two consecutive reconstructed images

        See https://www.aapm.org/pubs/ctprotocols/documents/ctterminologylexicon.pdf for further definitions of terms
        '''

        if kernel not in self.kernels:
            raise ValueError(f'{kernel} not in {self.kernels}')
        else:
            self.xcist.cfg.recon.kernelType = kernel

        slice_increment = slice_increment or slice_thickness

        if slice_increment:
            self.xcist.recon.sliceThickness = slice_increment
            # XCIST is mislabeled, recon.sliceThickness gives slice increment

        if mu_water:
            self.xcist.cfg.recon.mu = mu_water
        else:
            if self.xcist.physics.monochromatic != -1:
                self.xcist.recon.mu = xc.GetMu(
                    'water', self.xcist.physics.monochromatic)[0]/10
        if fov:
            self.xcist.cfg.recon.fov = fov
        print(f'fov size: {self.xcist.cfg.recon.fov}')

        self.xcist.cfg.recon.displayImagePictures = False

        if self.pitch > 0:
            self.recon = self.helical_recon()
        else:
            self.recon = self.axial_recon()
        self.projections = get_projection_data(self.xcist)
        self.groundtruth = None
        self.I0 = self.xcist.cfg.protocol.mA
        self.nsims = 1

        recons = []
        slice_increment = int(slice_increment) if slice_increment else slice_thickness
        slice_thickness = int(slice_thickness) if slice_thickness else slice_increment
        if not (slice_increment or slice_thickness):
            return self

        self.slice_thickness = slice_thickness or self.xcist.cfg.recon.sliceThickness
        self.slice_increment = slice_increment or self.slice_thickness
        n_slices = len(self.recon)
        starts = np.arange(0, n_slices, slice_increment, dtype=int)
        for slab_start in starts:
            recons.append(self.recon[slab_start:slab_start+slice_thickness].mean(axis=0))
        self.recon = np.stack(recons)

        return self

    def axial_recon(self):
        'performs axial recon from last acquired scan'
        self.xcist.cfg.recon.reconType = 'fdk_equiAngle'
        scan_width = 0.8*self.nominal_aperature
        valid_slices = int(scan_width // self.xcist.recon.sliceThickness)
        self.xcist.cfg.recon.sliceCount = valid_slices
        recons = []
        for proj in self._projections:
            self.xcist.cfg.resultsName = proj
            self.xcist.resultsName = self.xcist.cfg.resultsName
            vol = recon.recon_direct(self.xcist.cfg).transpose(2, 0, 1)
            recons.append(vol)
        return np.concatenate(recons, axis=0)

    def helical_recon(self):
        'performs axial recon from last acquired scan'
        if not self.pitch:
            raise ValueError(f'Helical recon requires scan data with pitch>0, detected pitch: {self.pitch}')
        self.xcist.cfg.recon.reconType = 'helical_equiAngle'
        exam_range = self.endZ - self.startZ
        sliceCount = int(exam_range / self.xcist.cfg.recon.sliceThickness)
        self.xcist.cfg.recon.sliceCount = sliceCount
        return recon.recon_direct(self.xcist.cfg).transpose(2, 0, 1)

    def write_to_dicom(self, fname: str | Path,
                       groundtruth=False) -> list[Path]:
        """
        write ct data to DICOM file, returns list of written dicom file names

        :param fname: filename to save image to (preferably with '.dcm`
            or related extension)
        :param groundtruth: Optional, whether to save the ground truth
            phantom image (no noise, blurring, or other artifacts).
            If True, 'self.groundtruth` is saved, if False (default)
            `self.recon` which contains blurring (and noise if 'add_noise`True)
        :returns: list[Path]

        Adapted from <https://github.com/DIDSR/pediatricIQphantoms/blob/main/src/pediatricIQphantoms/make_phantoms.py#L144>
        """
        fpath = pydicom.data.get_testdata_file("CT_small.dcm")
        ds = pydicom.dcmread(fpath)
        # update meta info
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
        del ds.PatientWeight
        del ds.ContrastBolusRoute
        del ds.ContrastBolusAgent
        ds.ImageComments =\
            f"effctive diameter [cm]: {self.patient_diameter/10}"
        ds.ScanOptions = self.xcist.cfg.protocol.scanTrajectory.upper()
        ds.ReconstructionDiameter = self.xcist.cfg.recon.fov
        ds.ConvolutionKernel = self.xcist.cfg.recon.kernelType
        ds.Exposure = self.xcist.cfg.protocol.mA

        # load image data
        ds.StudyDescription = f"{self.I0} photons " + self.seriesname +\
            " " + ds.ConvolutionKernel + self.xcist.cfg.recon.reconType
        if self.recon.ndim == 2:
            self.recon = self.recon[None]
        nslices, ds.Rows, ds.Columns = self.recon.shape

        ds.SpacingBetweenSlices = ds.SliceThickness
        ds.DistanceSourceToDetector = self.xcist.cfg.scanner.sdd
        ds.DistanceSourceToPatient = self.xcist.cfg.scanner.sid

        ds.PixelSpacing = [self.xcist.cfg.recon.fov/self.xcist.cfg.recon.imageSize,
                           self.xcist.cfg.recon.fov/self.xcist.cfg.recon.imageSize]
        ds.SliceThickness = self.xcist.cfg.recon.sliceThickness

        ds.KVP = self.kVp
        ds.StudyID = str(self.studyid)
        # series instance uid unique for each series
        end = ds.SeriesInstanceUID.split('.')[-1]
        new_end = str(int(end) + self.studyid)
        ds.SeriesInstanceUID = ds.SeriesInstanceUID.replace(end, new_end)

        # study instance uid unique for each series
        end = ds.StudyInstanceUID.split('.')[-1]
        new_end = str(int(end) + self.studyid)
        ds.StudyInstanceUID = ds.StudyInstanceUID.replace(end, new_end)
        ds.AcquisitionNumber = self.studyid

        fname = Path(fname)
        fname.parent.mkdir(exist_ok=True, parents=True)
        # saveout slices as individual dicom files
        fnames = []
        vol = self.groundtruth if groundtruth else self.recon
        if vol.ndim == 2:
            vol = vol[None]
        for slice_idx, array_slice in enumerate(vol):
            ds.InstanceNumber = slice_idx + 1  # image number
            # SOP instance UID changes every slice
            end = ds.SOPInstanceUID.split('.')[-1]
            new_end = str(int(end) + slice_idx + self.studyid + self.seriesid)
            ds.SOPInstanceUID = ds.SOPInstanceUID.replace(end, new_end)
            # MediaStorageSOPInstanceUID changes every slice
            end = ds.file_meta.MediaStorageSOPInstanceUID.split('.')[-1]
            new_end = str(int(end) + slice_idx + self.studyid + self.seriesid)
            ds.file_meta.MediaStorageSOPInstanceUID =\
                ds.file_meta.MediaStorageSOPInstanceUID.replace(end, new_end)
            # slice location and image position changes every slice
            ds.SliceLocation = self.nsims//2*ds.SliceThickness +\
                slice_idx*ds.SliceThickness
            ds.ImagePositionPatient[-1] = ds.SliceLocation
            ds.ImagePositionPatient[0] = -ds.Rows//2*ds.PixelSpacing[0]
            ds.ImagePositionPatient[1] = -ds.Columns//2*ds.PixelSpacing[1]
            ds.ImagePositionPatient[2] = ds.SliceLocation
            ds.PixelData = (array_slice.copy(order='C').astype('int16') -\
                int(ds.RescaleIntercept)).tobytes()
            dcm_fname = fname.parent /\
                f'{fname.stem}_{slice_idx:03d}{fname.suffix}'\
                if nslices > 1 else fname
            fnames.append(dcm_fname)
            pydicom.dcmwrite(dcm_fname, ds)
        return fnames
