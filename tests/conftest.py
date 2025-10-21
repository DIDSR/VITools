import pytest
import pydicom
import numpy as np

@pytest.fixture(scope="session")
def ctp404_dcm_path(tmp_path_factory):
    """Creates a dummy DICOM file for testing."""
    dcm_path = tmp_path_factory.mktemp("data") / "CTP404_groundtruth.dcm"
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.ImplementationClassUID = "1.2.3.4"
    file_meta.TransferSyntaxUID = "1.2.840.10008.1.2"
    ds = pydicom.dataset.FileDataset(dcm_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PixelData = np.zeros((10, 10), dtype=np.uint16).tobytes()
    ds.Rows, ds.Columns = 10, 10
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.RescaleIntercept = 0
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    pydicom.dcmwrite(dcm_path, ds)
    return dcm_path