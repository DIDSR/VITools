"""Tests for the Scanner class and related low-level CT simulation functionality.

This module contains a series of tests that verify the core functionalities
of the `Scanner` class, which wraps the XCIST CT simulation framework.
The tests cover:
- Basic simulation runs and output shapes.
- Loading of default and custom scanner configurations.
- Correctness of reconstruction dimensions based on scan parameters.
"""
from pathlib import Path
from shutil import rmtree

import numpy as np

from VITools import read_dicom, Scanner, Phantom
from VITools.examples import WaterPhantom

from utils import create_circle_phantom

test_dir = Path(__file__).parent.absolute()

# Create a standard phantom for use in multiple tests
circles = [-900, -300, -30, 30, 45, 200]
img = create_circle_phantom(image_size=200,
                            large_circle_value=0,
                            bg_value=-1000,
                            small_circle_values=circles,
                            num_small_circles=len(circles))
phantom = Phantom(img[None], spacings=(200, 1, 1))


def get_effective_diameter(ground_truth_mu: np.ndarray, pixel_width_mm: float) -> float:
    """Calculates the effective diameter based on AAPM TG204.

    The effective diameter is defined as 2 * sqrt(A/pi), where A is the
    cross-sectional area of the attenuating object.

    Args:
        ground_truth_mu (np.ndarray): A 2D array of HU values for a single slice.
        pixel_width_mm (float): The width of a single pixel in mm.

    Returns:
        float: The effective diameter of the object in the slice in mm.
    """
    A = np.sum(ground_truth_mu > -1000) * pixel_width_mm**2
    return 2 * np.sqrt(A / np.pi)


def scan_CTP404(test_dir: Path, views: int = 100, thickness: int = 1, increment: int = 1) -> Scanner:
    """Runs a standardized scan on the CTP404 phantom.

    This helper function sets up a `Scanner` with a cylinder phantom,
    runs a simulation, and performs reconstruction. It's used as a consistent
    basis for several tests.

    Args:
        test_dir (Path): The directory containing the test data.
        views (int, optional): The number of projection views to simulate.
            Defaults to 100.
        thickness (int, optional): The slice thickness for reconstruction (mm).
            Defaults to 1.
        increment (int, optional): The slice increment for reconstruction (mm).
            Defaults to 1.

    Returns:
        Scanner: The `Scanner` object after the scan and reconstruction
                 have been completed.
    """
    result_dir = test_dir / 'test_result'
    if result_dir.exists():
        rmtree(result_dir)

    phantom = WaterPhantom()
    ct = Scanner(phantom, 'Siemens_DefinitionFlash', output_dir=result_dir)
    ct.run_scan(views=views, startZ=-4, endZ=4)
    ct.run_recon(slice_thickness=thickness, slice_increment=increment)
    return ct


def test_scan_shape():
    """Tests the output shapes of a basic XCIST simulation.

    Verifies that the reconstructed volume, projection data, and the number
    of generated DICOM files have the expected dimensions after a standard scan.
    """
    views = 100
    ct = scan_CTP404(test_dir, views, increment=7)
    dcms = ct.write_to_dicom(ct.output_dir / 'test.dcm')
    dcms_in_dir = list(ct.output_dir.glob('*.dcm'))
    assert ct.recon.mean() > -800
    assert ct.recon.shape == (1, 512, 512)
    assert ct.projections.shape == (views,
                                    ct.xcist.cfg.scanner.detectorRowCount,
                                    ct.xcist.cfg.scanner.detectorColCount)
    assert len(dcms) == ct.recon.shape[0]
    assert sorted(dcms_in_dir) == sorted(dcms)


def test_load_scanner_config():
    """Tests the ability to load a different scanner configuration.

    Verifies that loading a new scanner configuration correctly updates the
    underlying simulation parameters, such as SID and collimation.
    """
    scanner = Scanner(phantom, 'Siemens_DefinitionFlash')
    original_sid = scanner.xcist.cfg.scanner.sid
    original_collimation = scanner.nominal_aperature
    scanner.load_scanner_config(test_dir / 'Scanner_Test')
    new_sid = scanner.xcist.cfg.scanner.sid
    new_collimation = scanner.nominal_aperature
    assert new_sid != original_sid
    assert new_collimation != original_collimation


def test_load_custom_scanner():
    """Tests initializing the Scanner with a path to a custom configuration.

    Ensures that the `Scanner` can be initialized directly with a path to a
    custom scanner configuration directory, not just a default name.
    """
    custom_scanner = test_dir / 'Scanner_Test'
    scanner = Scanner(phantom, scanner_model=str(custom_scanner))
    assert scanner.xcist.cfg.scanner.sid == 42


def test_recon_length():
    """Tests that the reconstructed volume length matches the scan range.

    Verifies that the number of slices in the reconstructed volume scales
    correctly with the specified axial scan length (startZ and endZ).
    """
    scanner = Scanner(phantom)

    center = 0
    width = scanner.nominal_aperature
    scanner.run_scan(startZ=center - width / 2, endZ=center + width / 2, views=10)
    scanner.run_recon(slice_thickness=1, slice_increment=1)
    assert len(scanner.recon) == 7, "Recon length should be 7 for single aperture scan"

    width = 2 * scanner.nominal_aperature
    scanner.run_scan(startZ=center - width / 2, endZ=center + width / 2, views=10)
    scanner.run_recon(slice_thickness=1, slice_increment=1)
    assert len(scanner.recon) == 14, "Recon length should be 14 for double aperture scan"

