# SPDX-License-Identifier: BSD-3-Clause
import re

####################################################################################################################
#
# Regex patterns for SHARAD, MARSIS, and LRS filenames
#
####################################################################################################################
patterns = {
    "SHARAD_EDR": re.compile(r"^e_(\d{7})_(\d{3})_((?:ss|ro)\d{1,2})_(\d{3})_([a-z])(?:_([sa]))?\.(dat|lbl)$",
                             re.IGNORECASE),
    "SHARAD_RDR": re.compile(r"^r_(\d{7})_(\d{3})_((?:ss|ro)\d{1,2})_(\d{3})_([a-z])\.(dat|lbl)$",
                             re.IGNORECASE),
    "SHARAD_US_EDR": re.compile(
    r"^OBS_(\d{6,9})000_(\d)_"                                     # Orbit + version
    r"(?:(Aux)_(\d{3})|"                                         # Aux file
    r"(Header)|"                                                 # Header file
    r"(HK)|"                                                     # Housekeeping
    r"(Log)|"                                                    # Log
    r"(Mode)_(\d{3})|"                                           # Mode file
    r"(Orbit)_(\d)|"                                             # Orbit file
    r"(Orbit)_(\d)_Log)"                                         # Orbit Log file
    r"\.(csv|txt|dat)$",                                         # <== Final group is extension
    re.IGNORECASE
    ),
    "SHARAD_US_RDR": re.compile(r"^s_(\d{8})_(rgram|geom)\.(lbl|tab|img)$", re.IGNORECASE),
    "SHARAD_US_SCS": re.compile(r"^s_(\d{8})_(emap|rtrn|sim)\.(img|csv)$", re.IGNORECASE),
    "SHARAD_UPB": re.compile(r'^UPB_(\d{9,11})_\d+_\d{2}_([A-Za-z0-9]+)_\d{3}\.raw$', re.IGNORECASE),
    "SHARAD_FPB": re.compile(r'^FPB_(\d{9,11})_\d+_\d{2}_([A-Za-z0-9]+)_\d{3}\.(tab|img)$', re.IGNORECASE),
    "SHARAD_FPB_SIM": re.compile(r'^FPB_(\d{9,11})_\d+_\d{2}_[A-Za-z]+_\d{3}_([A-Za-z]+)\.(csv|img)$', re.IGNORECASE),
    "SHARAD_QDA": re.compile(r'^QDA_(\d{9,11})_\d+_\d{2}_[A-Za-z]+_\d{3}\.dat\.(etm|mlk)$', re.IGNORECASE),
    "MARSIS": re.compile(r"^([er])_(\d{5})_(CAL|RXO|AIS|SS[1-5])_(ACQ|TRK)_(RAW|IND|UNC|CMP)_([MPT])(?:_([FG]))?\.(dat|lbl)$", re.IGNORECASE),
    "MARSIS_AIS_CAL_RXO": re.compile(
    r"^([e])_(\d{5})_(AIS|CAL|RXO)_([MPT])(?:_([FG]))?\.(dat|lbl)$",
    re.IGNORECASE
    ),
    "LRS_EDR": re.compile(r"^LRS_(SW|SA)_WF_(\d{2}[NS])_(\d{6}E)\.(lbl|tbl)$", re.IGNORECASE),
    "LRS_EDR_MALFORMED": re.compile(r"^LRS_(SW|SA)_WF_(\d{2}[NS])_(\d{7})\.(lbl|tbl)$", re.IGNORECASE),
    "LRS_RDR_COMPLEX": re.compile(r"^LRS_SAR(\d{2})KM_C_(\d{2}[NS])_(\d{6}E)\.(TBL|LBL)$", re.IGNORECASE),
    "LRS_RDR_POWER": re.compile(r"^LRS_SAR(\d{2})KM_(\d{14})\.(IMG|LBL)$", re.IGNORECASE),
}