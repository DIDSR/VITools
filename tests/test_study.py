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
from VITools import Study, get_available_phantoms

test_dir = Path(__file__).parent.absolute()


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
    input_df = Study.generate_from_distributions(phantoms=list(get_available_phantoms().keys()),
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