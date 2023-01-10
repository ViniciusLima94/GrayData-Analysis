import xarray as xr
import os
import numpy as np

class loader:
    def __init__(self, _ROOT: str = None):
        """
        Class used to load data.

        Inputs:
        ------
        _ROOT: str | None
            Root directory where to search the data.

        """
        self._ROOT = _ROOT

    def load_power(
        self,
        session: int = None,
        trial_type: int = 1,
        behavioral_response=1,
        aligned_at: str = "cue",
        channel_numbers: bool = False,
        monkey: str = "lucy",
    ):
        """
        Load file containing the power time series

        Inputs:
        ------

        session: int | None
            The sessions number.

        trial_type: int | None
            The type of the trial to load.

        behavioral_response: int | None
            The behavioral response outcome.

        aligned_at: str | "cue"
            The trial alignmend used when the LFP data was loaded.

        channel_numbers: bool | False
            Wheter to use the channel number with the roi name.

        monkey: str | "lucy"
            The monkey for which to load the data.

        Returns:
        -------
        power: xr.DataArray
            The Data Array containing the power data
        """
        # Path to the power file
        _RESULTS = os.path.join("Results", monkey, session, "session01")
        # Name of the power file
        power_file = self.__return_power_file_name(
            trial_type, behavioral_response, aligned_at
        )
        # Load power data
        power = xr.load_dataarray(os.path.join(self._ROOT, _RESULTS, power_file))
        power = power.transpose("roi", "freqs", "trials", "times")
        # Add channel number to roi name if needed
        if channel_numbers:
            sources, targets = np.tril_indices(power.sizes["roi"], k=-1)

            roi = [
                f"{r} ({ch})"
                for r, ch in zip(power.roi.data, power.attrs["channels_labels"])
            ]

            power = power.assign_coords({"roi": roi})

        return power

    def average_stages(
        self,
        data,
        stats: str,
        early_cue: float,
        early_delay: float,
    ):
        """
        Average it for each task stage.

        Inputs:
        ------

        data: array_like:
            Array containing the data.

        stats: str
            Which statistics to compute. Can be "mean",
            "95p" or "cv".

        early_cue: float
            Early cue used.

        early_delay: float
            Early delay used.

        Returns:
        -------
        out: xr.DataArray
            The Data Array averaged by stage
        """

        from GDa.util import create_stages_time_grid

        out = []
        # Creates stage mask
        attrs = data.attrs
        mask = create_stages_time_grid(
            attrs["t_cue_on"] - early_cue,
            attrs["t_cue_off"],
            attrs["t_match_on"],
            attrs["fsample"],
            data.times.data,
            data.sizes["trials"],
            early_delay=early_delay,
            align_to="cue",
            flatten=True,
        )
        for stage in mask.keys():
            mask[stage] = xr.DataArray(mask[stage], dims=("observations"))

        data = data.stack(observations=("trials", "times"))
        for stage in mask.keys():
            aux = data.isel(observations=mask[stage])
            if stats == "mean":
                out += [aux.mean("observations", skipna=True)]
            elif stats == "95p":
                out += [aux.quantile(0.95, "observations", skipna=True)]
            elif stats == "cv":
                mu = aux.mean("observations", skipna=True)
                sig = aux.std("observations", skipna=True)
                out += [sig / mu]

        out = xr.concat(out, "times")
        out.attrs = attrs
        return out

    def apply_min_rois(self, data: list, min_rois: int):
        """
        Concatenate data from different sessions and remove areas
        that have less than a minimum number of registers.

        Inputs:
        ------
        data: list
            List with DataArrays objects of recorgings from each session.
        min_rois: int
            Minimum number of registers required.

        Returns:
        -------
        data:
            DataArray of data of all sessions concatenated and
            without areas with less than 'min_rois' recordings.
        """
        # Concatenate channels
        data = xr.concat(data, dim="roi")
        # Get unique rois
        urois, counts = np.unique(data.roi.data, return_counts=True)
        # Get unique rois that has at leats min_rois channels
        urois = urois[counts >= min_rois]
        # Average channels withn the same roi
        data = data.groupby("roi").mean("roi", skipna=True)
        data = data.sel(roi=urois)
        return data

    def __return_power_file_name(self, trial_type, behavioral_response, aligned_at):
        """
        Return the name of the file containing the power time series.
        """
        _name = f"power_tt_{trial_type}_br_{behavioral_response}_at_{aligned_at}.nc"
        return _name
