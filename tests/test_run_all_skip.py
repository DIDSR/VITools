
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
    # Old logic would fail to read this and retry. New logic should see it exists and skip.
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

def test_run_all_skips_completed(setup_study_env):
    env = setup_study_env

    study = Study()
    study.metadata = env.metadata

    # Mock run_study to avoid actual simulation and track calls
    mock_results = pd.DataFrame({'image_file_path': ['path/to/img']})

    with patch.object(study, 'run_study', return_value=mock_results) as mock_run:
        # Mocking run calls
        study.run_all(parallel=False, overwrite=False)

        calls = mock_run.call_args_list
        called_ids = [args[0] for args, kwargs in calls]

        print(f"Called IDs: {called_ids}")

        assert 0 not in called_ids, "Case 0 should have been skipped"
        assert 1 not in called_ids, "Case 1 should have been skipped"
        assert 3 in called_ids, "Case 3 should have been run"
        assert 4 not in called_ids, "Case 4 (empty metadata) should have been skipped by file existence check"
        assert len(calls) == 1, f"Expected exactly 1 call (case 3), got {len(calls)}: {called_ids}"
