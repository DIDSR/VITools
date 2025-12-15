
import os
import shutil
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from VITools.study import Study

@pytest.fixture
def setup_study_env(tmp_path):
    # Setup directory structure
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # case_0000: completed
    (results_dir / "case_0000").mkdir()
    (results_dir / "case_0000" / "dicoms").mkdir()
    with open(results_dir / "case_0000" / "metadata_0.csv", "w") as f:
        f.write("case_id,some_data\ncase_0000,data")

    # case_0001: completed
    (results_dir / "case_0001").mkdir()
    (results_dir / "case_0001" / "dicoms").mkdir()
    (results_dir / "case_0001" / "lesion_masks").mkdir()
    with open(results_dir / "case_0001" / "metadata_1.csv", "w") as f:
        f.write("case_id,some_data\ncase_0001,data")

    # case_0003: incomplete
    (results_dir / "case_0003").mkdir()
    (results_dir / "case_0003" / "Vessel MIDA Head").mkdir()
    # No metadata file

    # case_0004: corrupted/empty metadata file
    (results_dir / "case_0004").mkdir()
    (results_dir / "case_0004" / "dicoms").mkdir()
    with open(results_dir / "case_0004" / "metadata_4.csv", "w") as f:
        f.write("") # Empty file

    # Create study metadata
    metadata = pd.DataFrame({
        'case_id': ['case_0000', 'case_0001', 'case_0003', 'case_0004'],
        'output_directory': [
            str(results_dir / "case_0000"),
            str(results_dir / "case_0001"),
            str(results_dir / "case_0003"),
            str(results_dir / "case_0004")
        ],
        'phantom': ['P1', 'P1', 'P1', 'P1'],
        'scanner_model': ['S1', 'S1', 'S1', 'S1'],
        'remove_raw': [False, False, False, False],
        'mA': [100, 100, 100, 100],
        'kVp': [120, 120, 120, 120],
        'views': [1000, 1000, 1000, 1000],
        'pitch': [0, 0, 0, 0],
        'fov': [250, 250, 250, 250],
        'scan_coverage': ['dynamic', 'dynamic', 'dynamic', 'dynamic'],
        'recon_kernel': ['soft', 'soft', 'soft', 'soft'],
        'slice_thickness': [1, 1, 1, 1],
        'slice_increment': [1, 1, 1, 1],
        'seed': [None, None, None, None]
    })

    return study_env(metadata, results_dir)

class study_env:
    def __init__(self, metadata, results_dir):
        self.metadata = metadata
        self.results_dir = results_dir

def test_run_all_skips_completed_serial(setup_study_env):
    """Verifies that run_all skips completed cases in serial mode."""
    env = setup_study_env

    study = Study()
    study.metadata = env.metadata

    mock_results = pd.DataFrame({'image_file_path': ['path/to/img']})

    # Ensure parallel=False and SLURM_ARRAY_TASK_ID is unset
    with patch.object(study, 'run_study', return_value=mock_results) as mock_run:
        with patch.dict(os.environ, {}, clear=True):
            study.run_all(parallel=False, overwrite=False)

            calls = mock_run.call_args_list
            called_ids = [args[0] for args, kwargs in calls]

            assert 0 not in called_ids
            assert 1 not in called_ids
            assert 3 in called_ids
            assert 4 not in called_ids

def test_run_all_parallel_worker_skips_completed(setup_study_env):
    """Verifies that a parallel worker (simulated by SLURM_ARRAY_TASK_ID) skips if the case is completed."""
    env = setup_study_env

    study = Study()
    study.metadata = env.metadata

    mock_results = pd.DataFrame({'image_file_path': ['path/to/img']})

    # Case 0 is completed. Worker assigned task 0 should skip.
    with patch.object(study, 'run_study', return_value=mock_results) as mock_run:
        with patch.dict(os.environ, {'SLURM_ARRAY_TASK_ID': '0'}):
            study.run_all(parallel=True, overwrite=False)
            assert mock_run.call_count == 0

def test_run_all_parallel_worker_runs_incomplete(setup_study_env):
    """Verifies that a parallel worker runs if the case is incomplete."""
    env = setup_study_env

    study = Study()
    study.metadata = env.metadata

    mock_results = pd.DataFrame({'image_file_path': ['path/to/img']})

    # Case 3 is incomplete. Worker assigned task 3 should run.
    with patch.object(study, 'run_study', return_value=mock_results) as mock_run:
        with patch.dict(os.environ, {'SLURM_ARRAY_TASK_ID': '3'}):
            study.run_all(parallel=True, overwrite=False)

            assert mock_run.call_count == 1
            assert mock_run.call_args[0][0] == 3
