import numpy as np
from scipy.signal import hilbert
from mne.filter import filter_data


def compute_plv_matrix(
    raw,
    low_band,
    high_band,
    start,
    end
):
    # === Compute PLV matrix for all channel pairs ===

    fs = float(raw.info["sfreq"])

    data = raw.get_data()           # shape: (n_channels, n_samples)
    n_channels = data.shape[0]

    # Prepare PLV matrix
    plv_matrix = np.zeros((n_channels, n_channels), dtype=float)

    # Two loops over all channel couples (i,j)
    for i in range(n_channels):
        # Take channel i as the low-frequency phase channel
        ch_lf = data[i, :]

        # ---- low-band filter + Hilbert  ----
        lf_filtered = filter_data(
            ch_lf, sfreq=fs, l_freq=low_band[0], h_freq=low_band[1]
        )
        analytic_lf = hilbert(lf_filtered)
        phase_lf = np.angle(analytic_lf)

        for j in range(n_channels):
            # Take channel j as the high-frequency amplitude channel
            ch_hf = data[j, :]

            # ---- high-band filter + Hilbert ----
            hf_filtered = filter_data(
                ch_hf, sfreq=fs, l_freq=high_band[0], h_freq=high_band[1]
            )
            analytic_hf = hilbert(hf_filtered)
            amp_hf = np.abs(analytic_hf)

            analytic_hf_abs = hilbert(amp_hf)
            phase_hf = np.angle(analytic_hf_abs)

            # ---- PLV (restricted to [start:end]) ----
            phase_diff = phase_lf - phase_hf
            plv_val = np.abs(
                np.mean(np.exp(1j * phase_diff[start:end]))
            )

            # store in matrix
            plv_matrix[i, j] = plv_val

    return plv_matrix


import numpy as np

def compute_plv_by_condition(
    raw,
    events,
    trials_dict,
    low_band: tuple[float, float],
    high_band: tuple[float, float],
    compute_plv_matrix,
):
    """
    Compute average PLV matrices for multiple trial conditions.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw iEEG data.
    events : pandas.DataFrame
        Events table with 'begSample' and 'endSample' columns.
    trials_dict : dict[str, iterable]
        Mapping from condition name -> iterable of trial indices.
        Example: {'easy': easy_trials, 'medium': medium_trials, 'hard': hard_trials}
    low_band : (float, float)
        Low-frequency band (Hz).
    high_band : (float, float)
        High-frequency band (Hz).
    compute_plv_matrix : callable
        Function with signature:
        compute_plv_matrix(raw, low_band, high_band, start, end) -> (n_ch, n_ch)

    Returns
    -------
    plv_dict : dict[str, np.ndarray]
        Mapping from condition name -> averaged PLV matrix (n_ch, n_ch).
    """

    n_channels = len(raw.ch_names)
    plv_dict = {}

    for cond_name, trials in trials_dict.items():
        if len(trials) == 0:
            raise ValueError(f"No trials provided for condition '{cond_name}'")

        plv = np.zeros((n_channels, n_channels), dtype=float)

        for t in trials:
            start = int(events.loc[t, "begSample"])
            end   = int(events.loc[t, "endSample"])

            plv += compute_plv_matrix(
                raw,
                low_band,
                high_band,
                start,
                end
            )

        plv /= len(trials)
        plv_dict[cond_name] = plv

    return plv_dict