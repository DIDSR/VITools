"""Tests for the Study class and its associated workflows.

This module verifies the functionality of the `Study` class, ensuring that it
can correctly:
- Generate study plans from distributions.
- Load study plans from DataFrames and CSV files.
- Run simulations based on the study plan.
- Append new scans to an existing study.
- Produce consistent and repeatable results.
"""
from pathlib import Path
from shutil import rmtree
from VITools import Study, get_available_phantoms
import pytest
import pandas as pd
import numpy as np
from VITools.study import Study
from VITools.phantom import Phantom
import shutil

test_dir = Path('tests/results').absolute()
if test_dir.exists():
    rmtree(test_dir)
test_dir.mkdir(parents=True)


def test_study():
    """Tests the core functionalities of the Study class.

    This test verifies several key features of the `Study` class:
    1.  It generates a study plan using `generate_from_distributions`.
    2.  It runs the study from the generated DataFrame.
    3.  It saves the DataFrame to a CSV and runs the study again from the file,
        ensuring the results are identical.
    4.  It tests the `append` method by adding the DataFrame and a single row
        to a new study instance and runs it.
    5.  It checks for equality between the results of the first two runs.
    """
    from VITools.phantom import Phantom
    import numpy as np
    phantoms = get_available_phantoms()
    input_df = Study.generate_from_distributions(phantoms=['Water Phantom'],
                                                 study_count=2,
                                                 views=[20],
                                                 scan_coverage=(0, 7))
    study1 = Study(input_df)
    study1.run_all(parallel=False, overwrite=True)
    results1 = study1.results

    input_csv = test_dir / 'input.csv'
    input_df.to_csv(input_csv, index=False)

    study2 = Study(input_csv)
    study2.run_all(parallel=False, overwrite=True)
    results2 = study2.results

    study3 = Study()
    study3.append(input_df)
    study3.append(**input_df.iloc[0].to_dict())
    study3.run_all(parallel=False, overwrite=True)
    results3 = study3.results

    assert results1.equals(results2), "Results from DataFrame and CSV should be identical"
    assert len(results3) > len(results1), "Append should increase the number of results"
    # The exact length of results3 can be complex due to multi-slice phantoms,
    # so we just check that it's larger than the original.

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

def test_generate_from_distributions():
    """Test the generate_from_distributions method."""
    phantoms = ["Water Phantom"]
    df = Study.generate_from_distributions(phantoms, study_count=5, scan_coverage='dynamic')
    assert len(df) == 5
    assert df["scan_coverage"].iloc[0] == "dynamic"

    df = Study.generate_from_distributions(phantoms, study_count=2, scan_coverage=[0, 1])
    assert df["scan_coverage"].iloc[0] == [0.0, 1.0]

def test_append(output_dir):
    """Test the append method."""
    study = Study()
    assert len(study) == 0

    # Append a single scan
    study.append("Water Phantom", output_directory=output_dir)
    assert len(study) == 1

    # Append a DataFrame
    df = Study.generate_from_distributions(["Water Phantom"], study_count=2)
    study.append(df)
    assert len(study) == 3

def test_run_all_serial(output_dir):
    """Test the run_all method in serial mode."""
    # This test is slow, so we only run a single case
    df = Study.generate_from_distributions(["Water Phantom"], study_count=1, output_directory=output_dir, slice_increment=[7], views=[10], scan_coverage=[0, 1])
    study = Study(df)
    study.run_all(parallel=False, overwrite=True)
    assert len(study.results) == 1

def test_get_images(output_dir):
    """Test the get_images method."""
    df = Study.generate_from_distributions(["Water Phantom"], study_count=1, output_directory=output_dir, slice_increment=[7], views=[10], scan_coverage=[0, 1])
    study = Study(df)
    study.run_all(parallel=False, overwrite=True)
    images = study.get_images(patientid=0)
    assert images.shape == (1, 512, 512)