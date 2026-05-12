import numpy as np


class NISTMSConverter:
    """
    Convert floating-point mass spectra to NIST nominal-mass format.

    Parameters
    ----------
    rounding_mode : {'round', 'floor', 'ceil'}, default='round'
        How to convert m/z floats to integers:
        - 'round' → floor(x + 0.5)
        - 'floor' → floor(x)
        - 'ceil'  → ceil(x)
    threshold : float, default=0.01
        Relative intensity cutoff; peaks below (threshold * base_peak) are dropped.
    base_value : int, default=999
        Scale the base-peak intensity to this integer.
    keep_highest : bool, default=False
        If True, keep only the single highest-intensity float peak in each integer bin.
        If False, sum all intensities in each bin.
    """

    def __init__(self,
                 rounding_mode: str = 'round',
                 threshold: float = 0.01,
                 base_value: int = 999,
                 keep_highest: bool = False):
        modes = ('round', 'floor', 'ceil')
        if rounding_mode not in modes:
            raise ValueError(f"rounding_mode must be one of {modes}")
        self.rounding_mode = rounding_mode
        self.threshold = threshold
        self.base_value = base_value
        self.keep_highest = keep_highest

    def _to_integer_bins(self, mz: np.ndarray) -> np.ndarray:
        if self.rounding_mode == 'round':
            return np.floor(mz + 0.5).astype(int)
        elif self.rounding_mode == 'floor':
            return np.floor(mz).astype(int)
        else:  # 'ceil'
            return np.ceil(mz).astype(int)

    def convert(self,
                mz: np.ndarray,
                intensity: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Perform the full NIST nominal-mass conversion.

        Parameters
        ----------
        mz : np.ndarray (float)
            Measured m/z values.
        intensity : np.ndarray (float)
            Corresponding intensities.

        Returns
        -------
        nominal_mz : np.ndarray (int)
            Integer m/z bins after rounding and filtering.
        nominal_intensity : np.ndarray (int)
            Integer intensities normalized to base_value.
        """
        if mz.shape != intensity.shape:
            raise ValueError("`mz` and `intensity` must have the same shape")

        # 1. Round/truncate/ceil to integer bins
        bins = self._to_integer_bins(mz)

        # 2. Consolidate peaks in each bin
        if self.keep_highest:
            peak_dict = {}
            for b, i in zip(bins, intensity):
                peak_dict[b] = i if (b not in peak_dict or i > peak_dict[b]) else peak_dict[b]
            unique_bins = np.array(sorted(peak_dict.keys()), dtype=int)
            consolidated = np.array([peak_dict[b] for b in unique_bins], dtype=float)
        else:
            unique_bins, inv = np.unique(bins, return_inverse=True)
            consolidated = np.zeros_like(unique_bins, dtype=float)
            np.add.at(consolidated, inv, intensity)

        # 3. Threshold: drop low-intensity bins
        max_int = consolidated.max()
        if max_int == 0:
            return unique_bins, consolidated.astype(int)
        keep_mask = (consolidated / max_int) >= self.threshold
        filtered_bins = unique_bins[keep_mask]
        filtered_int = consolidated[keep_mask]

        # 4. Normalize to base_value, then round to int
        scaled = filtered_int / filtered_int.max() * self.base_value
        final_intensity = np.floor(scaled + 0.5).astype(int)

        # 5. Sort by m/z
        sort_idx = np.argsort(filtered_bins)
        nominal_mz = filtered_bins[sort_idx]
        nominal_intensity = final_intensity[sort_idx]

        return nominal_mz, nominal_intensity