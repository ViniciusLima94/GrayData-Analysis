import pandas as pd
import xarray as xr


# T-values
t_power = xr.load_dataarray(
    'tval_pow_1_br_1_aligned_cue_avg_1.nc').to_dataframe(name='power')
t_degree = xr.load_dataarray(
    't_degree_avg_1_thr_1.nc').to_dataframe(name='degree')
t_coreness = xr.load_dataarray(
    't_coreness_avg_1_thr_1.nc').to_dataframe(name='coreness')
t_efficiency = xr.load_dataarray(
    't_efficiency_avg_1_thr_1.nc').to_dataframe(name='efficiency')

# p-values
p_power = xr.load_dataarray(
    'pval_pow_1_br_1_aligned_cue_avg_1.nc').to_dataframe(name='power')
p_degree = xr.load_dataarray(
    'pval_degree_avg_1_thr_1.nc').to_dataframe(name='degree')
p_coreness = xr.load_dataarray(
    'pval_coreness_avg_1_thr_1.nc').to_dataframe(name='coreness')
p_efficiency = xr.load_dataarray(
    'pval_efficiency_avg_1_thr_1.nc').to_dataframe(name='efficiency')

power = t_power * (p_power <= 0.05)
degree = t_degree * (p_degree <= 0.05)
coreness = t_coreness * (p_coreness <= 0.05)
efficiency = t_efficiency * (p_efficiency <= 0.05)

df = pd.concat([power, degree, coreness, efficiency], axis=1)
df.to_csv("mi_df.csv")
