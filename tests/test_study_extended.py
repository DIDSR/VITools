import pytest
from pathlib import Path
import pandas as pd
from VITools.study import Study
from .utils import create_circle_phantom
from VITools.phantom import Phantom
import shutil

@pytest.fixture
def output_dir(tmp_path):
    return tmp_path

def test_repr():
    """Test the __repr__ method of the Study class."""
    study = Study()
    assert "Input metadata" in repr(study)
    assert "Results" in repr(study)

def test_clear_previous_results(output_dir):
    """Test the clear_previous_results method."""
    # Create a dummy output directory
    dummy_dir = output_dir / "case_0000"
    dummy_dir.mkdir()
    assert dummy_dir.exists()

    # Create a study with metadata pointing to the dummy directory
    metadata = pd.DataFrame({"output_directory": [str(dummy_dir)]})
    study = Study(metadata)

    # Clear the results and assert the directory is deleted
    study.clear_previous_results()
    assert not dummy_dir.exists()

@pytest.mark.usefixtures("register_test_phantom")
def test_generate_from_distributions():
    """Test the generate_from_distributions method."""
    phantoms = ["TestPhantom"]
    df = Study.generate_from_distributions(phantoms, study_count=5, scan_coverage='dynamic')
    assert len(df) == 5
    assert df["scan_coverage"].iloc[0] == "dynamic"

    df = Study.generate_from_distributions(phantoms, study_count=2, scan_coverage=[0, 1])
    assert df["scan_coverage"].iloc[0] == [0.0, 1.0]

@pytest.mark.usefixtures("register_test_phantom")
def test_append(output_dir):
    """Test the append method."""
    study = Study()
    assert len(study) == 0

    # Append a single scan
    study.append("TestPhantom", output_directory=output_dir)
    assert len(study) == 1

    # Append a DataFrame
    df = Study.generate_from_distributions(["TestPhantom"], study_count=2)
    study.append(df)
    assert len(study) == 3

@pytest.mark.usefixtures("register_test_phantom")
def test_run_all_serial(output_dir):
    """Test the run_all method in serial mode."""
    # This test is slow, so we only run a single case
    df = Study.generate_from_distributions(["TestPhantom"], study_count=1, output_directory=output_dir, slice_increment=[7], views=[10])
    study = Study(df)
    study.run_all(parallel=False, overwrite=True)
    assert len(study.results) == 1

@pytest.mark.usefixtures("register_test_phantom")
def test_get_images(output_dir):
    """Test the get_images method."""
    df = Study.generate_from_distributions(["TestPhantom"], study_count=1, output_directory=output_dir, slice_increment=[7], views=[10])
    study = Study(df)
    study.run_all(parallel=False, overwrite=True)
    images = study.get_images(patientid=0)
    assert images.shape == (1, 512, 512)