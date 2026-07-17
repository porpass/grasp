# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Iterator

import numpy as np

from ..grasp_types import RecordFormat, MARSISMode

auxlen = 215

SUBSYSTEM_MODES = {
    'SS1': MARSISMode(n_chan=2, n_filt=1, filt_str=["ZERO"], chan_str=["F1", "F2"]),
    'SS2': MARSISMode(n_chan=2, n_filt=1, filt_str=["ZERO"], chan_str=["F1", "F2"]),
    'SS3': MARSISMode(n_chan=2, n_filt=3, filt_str=["MINUS1", "ZERO", "PLUS1"], chan_str=["F1", "F2"]),
    'SS4': MARSISMode(n_chan=2, n_filt=5, filt_str=["MINUS2", "MINUS1", "ZERO", "PLUS1", "PLUS2"], chan_str=["F1", "F2"]),
    'SS5': MARSISMode(n_chan=2, n_filt=3, filt_str=["MINUS1", "ZERO", "PLUS1"], chan_str=["F1", "F2"]),
}

DCG_TO_FCEN = np.array([1.8e6, 3.0e6, 4.0e6, 5.0e6])

SYSTEM_DELAY = np.array([5.31e-6, 5.04e-6, 4.67e-6, 4.49e-6])

RECORD_FORMAT = {
    #
    # EDRs
    #
    ('EDR', 'CAL'): RecordFormat(reclen=313856, auxlen=auxlen),
    ('EDR', 'RXO'): RecordFormat(reclen=313856, auxlen=auxlen),
    ('EDR', 'AIS'): RecordFormat(reclen=25856, auxlen=auxlen),
    # SS1
    ('EDR', 'SS1', 'ACQ', 'CMP'): RecordFormat(reclen=4864, auxlen=auxlen),
    ('EDR', 'SS1', 'TRK', 'CMP'): RecordFormat(reclen=4864, auxlen=auxlen),
    # SS2
    ('EDR', 'SS2', 'ACQ', 'CMP'): RecordFormat(reclen=4864, auxlen=auxlen),
    ('EDR', 'SS2', 'TRK', 'CMP'): RecordFormat(reclen=2816, auxlen=auxlen),
    # SS3
    ('EDR', 'SS3', 'ACQ', 'CMP'): RecordFormat(reclen=4864, auxlen=auxlen),
    ('EDR', 'SS3', 'TRK', 'CMP'): RecordFormat(reclen=6912, auxlen=auxlen),
    ('EDR', 'SS3', 'TRK', 'RAW'): RecordFormat(reclen=2048, auxlen=auxlen),
    # SS4
    ('EDR', 'SS4', 'ACQ', 'CMP'): RecordFormat(reclen=2816, auxlen=auxlen),
    ('EDR', 'SS4', 'TRK', 'CMP'): RecordFormat(reclen=11008, auxlen=auxlen),
    # SS5
    #('EDR', 'SS5', 'ACQ', 'CMP'): RecordFormat(reclen=0, auxlen=auxlen), # Don't know if this mode has ever been used
    ('EDR', 'SS5', 'TRK', 'CMP'): RecordFormat(reclen=6912, auxlen=auxlen), # Don't know if this mode has ever been used
    #
    # RDRs
    #
    ('RDR', 'SS3', 'TRK', 'CMP'): RecordFormat(reclen=24823),
    ('RDR', 'SS3', 'TRK', 'RAW'): RecordFormat(reclen=15927),
}

RAW_OVERRIDES = {
    'FS': 2.8e6,
    'DT': 3.57e-7,
    'DZ': 53.534,
    'NSAMP': 980,
    'NFFT': 1024,
}


def iter_channel_filter(operative_mode: str) -> Iterator[tuple[str, str, str]]:
    """Iterate (filter_str, channel_str, sci_key) for a MARSIS operative mode.

    Every wrapper under ``grasp.marsis`` needs the same double loop over
    the operative mode's channel/filter grid. This helper centralises the
    iteration and the canonical ``ECHO_<F>_<C>_DIP`` science-key format
    so individual wrappers can write::

        for f_str, c_str, key in iter_channel_filter(operative_mode):
            ...

    instead of nested ``for _c in range(n_chan): for _f in range(n_filt):``
    boilerplate. Callers decide what to do when ``key`` is missing from
    their own data dict — silent skip (output writers) or raise KeyError
    (input processing stages).

    Args:
        operative_mode: MARSIS operative mode (e.g. ``"SS3"``).
            Case-insensitive.

    Yields:
        Tuples of ``(filter_str, channel_str, sci_key)`` where
        ``sci_key = f"ECHO_{filter_str}_{channel_str}_DIP"``.

    Raises:
        KeyError: If ``operative_mode`` is not a recognised
            MARSIS subsystem mode.
    """
    mode_info = SUBSYSTEM_MODES[operative_mode.upper()]
    for c in range(mode_info.n_chan):
        c_str = mode_info.chan_str[c]
        for f in range(mode_info.n_filt):
            f_str = mode_info.filt_str[f]
            yield f_str, c_str, f"ECHO_{f_str}_{c_str}_DIP"