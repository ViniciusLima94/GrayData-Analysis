import netchos
import xarray as xr
from frites.conn import conn_reshape_undirected

mi = xr.load_dataarray("Results/lucy/mi_pow_rfx/mi_coh_avg_1.nc")
p = xr.load_dataarray("Results/lucy/mi_pow_rfx/pval_coh_avg_1.nc")

mi_sig = mi * (p <= 0.05)

for f in range(mi_sig.sizes["freqs"]):
    c = conn_reshape_undirected(mi_sig.isel(freqs=f))
    for t in range(c.sizes["times"]):
        fig = netchos.circular(c.isel(times=t), fig=None)
        fig.write_image(f"figures/t_{t}_f_{f}.png")
