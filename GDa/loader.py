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
        decim: int = 20
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
        decim: int | 20
            The decimation factor used

        Returns:
        -------
        power: xr.DataArray
            The Data Array containing the power data
        """
        # Path to the power file
        _RESULTS = os.path.join("Results", monkey, session, "session01")
        # Name of the power file
        power_file = self.__return_power_file_name(
            trial_type, behavioral_response, aligned_at, decim
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


    def load_pecst(
        self,
        session: int = None,
        metric: float = "pec",
        aligned_at: str = "cue",
        monkey: str = "lucy",
    ):
        """
        Load file containing the power time series

        Inputs:
        ------

        session: int | None
            The sessions number.

        aligned_at: str | "cue"
            The trial alignmend used when the LFP data was loaded.

        channel_numbers: bool | False
            Wheter to use the channel number with the roi name.

        monkey: str | "lucy"
            The monkey for which to load the data.

        Returns:
        -------
        PEC ST: xr.DataArray
            The Data Array containing the power data
        """
        # Path to the power file
        _RESULTS = os.path.join("Results", monkey, session,
                                "session01", "network")
        # Name of the power file
        pec_file = self.__return_pecst_file_name(
            metric, aligned_at
        )
        # Load power data
        pecst = xr.load_dataarray(os.path.join(self._ROOT, _RESULTS, pec_file))
        pecst = pecst.transpose("roi", "freqs", "trials", "times")

        return pecst



    def load_co_crakcle(
        self,
        session: int = None,
        trial_type: int = 1,
        monkey: str = "lucy",
        strength: bool = False,
        thr: int = 90,
        incorrect: bool = False,
        rectf: int=None,
        surrogate: bool = False,
        drop_roi: str = None
    ):
        """
        Load file containing the power time series

        Inputs:
        ------

        session: int | None
            The sessions number.

        trial_type: int | None
            The type of the trial to load.

        monkey: str | "lucy"
            The monkey for which to load the data.

        strength: bool | False
            Whether to return the strength rather than the Kij matrix.

        Returns:
        -------
        power: xr.DataArray
            The Data Array containing the power data
        """
        # Path to the power file
        _RESULTS = os.path.join("Results", monkey, "crk_stats")
        # Name of the power file
        ttype = "task" if trial_type == 1 else "fix"

        if incorrect:
            prefix = f"kij_{ttype}_incorrect"
        else:
            prefix = f"kij_{ttype}"

        if not surrogate:
            if not isinstance(rectf, int):
                crk_file = f"{prefix}_{session}_q_{thr}.nc"
            else:
                crk_file = f"{prefix}_{session}_q_{thr}_rectf_{rectf}.nc"
        else:
            crk_file = f"{prefix}_surr_{session}_q_{thr}.nc"

        # Load power data
        kij = xr.load_dataarray(os.path.join(self._ROOT, _RESULTS, crk_file))

        A = kij.copy()
        if isinstance(drop_roi, str):
            assert drop_roi in ["same", "diff"]
            unique_rois = np.unique(kij.sources)
            for ur in unique_rois:
                idx = kij.sources.data == ur
                if drop_roi == "diff":
                    A[:, idx, np.logical_not(idx), :] = 0
                else:
                    A[:, idx, idx, :] = 0
        kij = A.copy()
        del A

        if not surrogate:
            kij = kij.transpose("sources", "targets", "freqs","times")
        else:
            kij = kij.transpose("sources", "targets", "freqs","times", "boot")
        if strength:
            kij = kij.mean("targets")
            kij = kij.rename({"sources": "roi"})
        return kij


    def load_burst_prob(self,
                        session: int = None,
                        trial_type: int = 1,
                        aligned_at: str = "cue",
                        monkey: str = "lucy",
                        thr: int = 90,
                        conditional: bool = False):
        """
        Load burst probability data from file and return as a xarray.DataArray.

        Inputs:
        ------
        session: int | None
            session number
        trial_type: int | None
            trial type number
        behavioral_response: int | None
            behavioral response outcome
        aligned_at: str | "cue"
            whether the data is aligned to the cue or the burst
        channel_numbers: bool | False
            whether to use the channel number with the roi name
        monkey: str | "lucy"
            name of the monkey

        Returns:
        -------
        power: xr.DataArray
            containing burst probability data
            with dimensions "roi", "freqs", "boot", and "times".
        """
        # Path to the power file
        _RESULTS = os.path.join("Results", monkey, "rate_modulations")
        # Name of the power file
        pb_file = self.__return_pb_file_name(
            trial_type, session, aligned_at, thr, conditional
        )
        # Load power data
        P_b = xr.load_dataarray(os.path.join(self._ROOT, _RESULTS, pb_file))
        if not conditional:
            return P_b.transpose("roi", "freqs", "boot", "times")
        return P_b.transpose("roi", "freqs", "boot", "stim", "times")

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


    def __return_pecst_file_name(self, metric, aligned_at):
        """
        Return the name of the file containing the power time series.
        """
        _name = f"{metric}_degree_at_{aligned_at}.nc"
        return _name


    def __return_power_file_name(self, trial_type, behavioral_response, aligned_at, decim):
        """
        Return the name of the file containing the power time series.
        """
        _name = f"power_tt_{trial_type}_br_{behavioral_response}_at_{aligned_at}_decim_{decim}.nc"
        return _name


    def __return_pb_file_name(self, trial_type, session, aligned_at, thr, conditional):
        """
        Return the name of the file containing the burst prob time series.
        """
        if trial_type == 1:
            sufix = "task"
        elif trial_type == 2:
            sufix = "fix"
        if not conditional: 
            return f"P_b_{sufix}_{session}_at_{aligned_at}_q_{thr}.nc"
        return f"P_b_{sufix}_stim_{session}_at_{aligned_at}_q_{thr}.nc"
