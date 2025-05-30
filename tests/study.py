# %%
from pathlib import Path
from VITools import Study

test_dir = Path(__file__).parent.absolute()

input_df = Study.generate_from_distributions(2, Views=[100], ScanCoverage=(0, 7))
input_df
# %%
study1 = Study(input_df)
study1
study1.run_all(parallel=False)
results1 = study1.results
# %%
input_csv = test_dir / 'input.csv'
input_df.to_csv(input_csv, index=False)

study2 = Study(input_csv)
study2.run_all(parallel=False)
results2 = study2.results
# %%
study3 = Study()
study3.append(**input_df.iloc[0])
study3.append(input_df)
study3.run_all(parallel=True)
results3 = study3.results
# %%
study1.results
# %%
assert study1.metadata.equals(study2.metadata)
assert study1.results.equals(study2.results)
# %%
study3.metadata
# %%
