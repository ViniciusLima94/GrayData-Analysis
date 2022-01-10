import pandas as pd
import xarray as xr


power = xr.load_dataarray(
    'tval_pow_1_br_1_aligned_cue_avg_1.nc').to_dataframe(name='power')
degree = xr.load_dataarray(
    't_degree_avg_1_thr_1.nc').to_dataframe(name='degree')
coreness = xr.load_dataarray(
    't_coreness_avg_1_thr_1.nc').to_dataframe(name='coreness')
efficiency = xr.load_dataarray(
    't_efficiency_avg_1_thr_1.nc').to_dataframe(name='efficiency')

df = pd.concat([power, degree, coreness, efficiency], axis=1)
df.to_csv("mi_df.csv")
