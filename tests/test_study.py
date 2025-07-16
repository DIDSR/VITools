from pathlib import Path
from VITools import Study, get_available_phantoms

test_dir = Path(__file__).parent.absolute()


def test_study():
    input_df = Study.generate_from_distributions(phantoms=get_available_phantoms(),
                                                 study_count=2,
                                                 views=[20],
                                                 scan_coverage=(0, 7))
    study1 = Study(input_df)
    study1.run_all(parallel=False)
    results1 = study1.results

    input_csv = test_dir / 'input.csv'
    input_df.to_csv(input_csv, index=False)

    study2 = Study(input_csv)
    study2.run_all(parallel=False)
    results2 = study2.results
    study3 = Study()
    study3.append(input_df)
    study3.append(**input_df.iloc[0])
    study3.run_all(parallel=False)
    results3 = study3.results

    assert results1.equals(results2)
    print(len(results1), len(results2), len(results3))  # eventually len(results3) == 2 * len(results1) + 7
