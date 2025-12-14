import pandas as pd
from pathlib import Path
import os
import pytest
from VITools.study import clean_study

def test_clean_study(tmp_path):
    # Setup
    study_root = tmp_path / "my_study"
    study_root.mkdir()

    # Define cases
    cases = ["case_0000", "case_0001", "case_0002"]
    output_directories = [study_root / c for c in cases]

    # Create metadata dataframe
    df = pd.DataFrame({
        "case_id": cases,
        "output_directory": [str(p) for p in output_directories],
        "phantom": ["X"] * 3,
        "scanner_model": ["Y"] * 3
        # Add other required columns if Study validates them (Study constructor doesn't seem to validate much)
    })

    # Save study plan
    study_plan_path = study_root / "my_study_plan.csv"
    df.to_csv(study_plan_path, index=False)

    # Create directories
    for p in output_directories:
        p.mkdir()

    # case_0000: empty (should be removed)

    # case_0001: empty (should be removed)

    # case_0002: completed
    # Needs a metadata_0002.csv inside
    # Study.get_scans_completed looks for metadata_*.csv
    # and expects to read it into a DF.
    meta_df = pd.DataFrame({
        "case_id": ["case_0002"],
        "some_result": [1.0]
    })
    meta_df.to_csv(output_directories[2] / "metadata_0002.csv", index=False)

    # Verify setup
    assert (study_root / "case_0000").exists()
    assert (study_root / "case_0001").exists()
    assert (study_root / "case_0002").exists()

    # Run clean_study
    clean_study(str(study_root))

    # Verify results
    assert not (study_root / "case_0000").exists(), "case_0000 should have been removed"
    assert not (study_root / "case_0001").exists(), "case_0001 should have been removed"
    assert (study_root / "case_0002").exists(), "case_0002 should have been kept"
    assert (study_root / "case_0002" / "metadata_0002.csv").exists()
