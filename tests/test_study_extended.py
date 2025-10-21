import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from VITools.study import Study
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

def test_generate_from_distributions(monkeypatch):
    """Test the generate_from_distributions method."""
    monkeypatch.setattr('VITools.study.get_available_phantoms', lambda: {'TestPhantom': lambda: Phantom(np.zeros((10,10,10)))})
    phantoms = ["TestPhantom"]
    df = Study.generate_from_distributions(phantoms, study_count=5, scan_coverage='dynamic')
    assert len(df) == 5
    assert df["scan_coverage"].iloc[0] == "dynamic"

    df = Study.generate_from_distributions(phantoms, study_count=2, scan_coverage=[0, 1])
    assert df["scan_coverage"].iloc[0] == [0.0, 1.0]

def test_append(output_dir, monkeypatch):
    """Test the append method."""
    monkeypatch.setattr('VITools.study.get_available_phantoms', lambda: {'TestPhantom': lambda: Phantom(np.zeros((10,10,10)))})
    study = Study()
    assert len(study) == 0

    # Append a single scan
    study.append("TestPhantom", output_directory=output_dir)
    assert len(study) == 1

    # Append a DataFrame
    df = Study.generate_from_distributions(["TestPhantom"], study_count=2)
    study.append(df)
    assert len(study) == 3

def test_run_all_serial(output_dir, monkeypatch):
    """Test the run_all method in serial mode."""
    monkeypatch.setattr('VITools.study.get_available_phantoms', lambda: {'TestPhantom': lambda: Phantom(np.zeros((10,10,10)))})
    # This test is slow, so we only run a single case
    df = Study.generate_from_distributions(["TestPhantom"], study_count=1, output_directory=output_dir, slice_increment=[7], views=[10])
    study = Study(df)
    study.run_all(parallel=False, overwrite=True)
    assert len(study.results) == 1

def test_get_images(output_dir, monkeypatch):
    """Test the get_images method."""
    monkeypatch.setattr('VITools.study.get_available_phantoms', lambda: {'TestPhantom': lambda: Phantom(np.zeros((10,10,10)))})
    df = Study.generate_from_distributions(["TestPhantom"], study_count=1, output_directory=output_dir, slice_increment=[7], views=[10])
    study = Study(df)
    study.run_all(parallel=False, overwrite=True)
    images = study.get_images(patientid=0)
    assert images.shape == (1, 512, 512)