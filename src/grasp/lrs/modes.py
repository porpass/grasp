# SPDX-License-Identifier: BSD-3-Clause
from ..grasp_types import LRSMode, RecordFormat

FLOAT = 1055
COMPL = 8055

SUBSYSTEM_MODES = {
    'SW_WF': LRSMode(prf=20.0, pri=1.0 / 20.0),
    'SA_WF': LRSMode(prf=2.5, pri=1.0 / 2.5),
}

RECORD_FORMAT = {
    #
    # EDRs
    #
    ('EDR',): RecordFormat(reclen=4155, auxlen=None),
    #
    # RDRs
    #
    ('RDR', 'SAR05KM'): RecordFormat(reclen=FLOAT, auxlen=None),
    ('RDR', 'SAR05KM-C'): RecordFormat(reclen=COMPL, auxlen=None),
    ('RDR', 'SAR10KM'): RecordFormat(reclen=FLOAT, auxlen=None),
    ('RDR', 'SAR10KM-C'): RecordFormat(reclen=COMPL, auxlen=None),
    ('RDR', 'SAR40KM'): RecordFormat(reclen=FLOAT, auxlen=None),

}