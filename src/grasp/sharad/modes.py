# SPDX-License-Identifier: BSD-3-Clause
from ..grasp_types import RecordFormat, SHARADMode


SCI8 = int(3786)
SCI6 = int(2886)
SCI4 = int(1986)
AUX = int(267)
RDR = int(5822)
USEDR = int(3600)
USRDR = int(14400)
QDA = int(16384)


RECORD_FORMAT = {
        #
        # EDRs
        #
        ('EDR', 'SS01'): RecordFormat(reclen=SCI8, auxlen=AUX),
        ('EDR', 'SS02'): RecordFormat(reclen=SCI6, auxlen=AUX),
        ('EDR', 'SS03'): RecordFormat(reclen=SCI4, auxlen=AUX),
        ('EDR', 'SS04'): RecordFormat(reclen=SCI8, auxlen=AUX),
        ('EDR', 'SS05'): RecordFormat(reclen=SCI6, auxlen=AUX),
        ('EDR', 'SS06'): RecordFormat(reclen=SCI4, auxlen=AUX),
        ('EDR', 'SS07'): RecordFormat(reclen=SCI8, auxlen=AUX),
        ('EDR', 'SS08'): RecordFormat(reclen=SCI6, auxlen=AUX),
        ('EDR', 'SS09'): RecordFormat(reclen=SCI4, auxlen=AUX),
        ('EDR', 'SS10'): RecordFormat(reclen=SCI8, auxlen=AUX),
        ('EDR', 'SS11'): RecordFormat(reclen=SCI6, auxlen=AUX),
        ('EDR', 'SS12'): RecordFormat(reclen=SCI4, auxlen=AUX),
        ('EDR', 'SS13'): RecordFormat(reclen=SCI8, auxlen=AUX),
        ('EDR', 'SS14'): RecordFormat(reclen=SCI6, auxlen=AUX),
        ('EDR', 'SS15'): RecordFormat(reclen=SCI4, auxlen=AUX),
        ('EDR', 'SS16'): RecordFormat(reclen=SCI8, auxlen=AUX),
        ('EDR', 'SS17'): RecordFormat(reclen=SCI6, auxlen=AUX),
        ('EDR', 'SS18'): RecordFormat(reclen=SCI4, auxlen=AUX),
        ('EDR', 'SS19'): RecordFormat(reclen=SCI8, auxlen=AUX),
        ('EDR', 'SS20'): RecordFormat(reclen=SCI6, auxlen=AUX),
        ('EDR', 'SS21'): RecordFormat(reclen=SCI4, auxlen=AUX),
        #
        # RDRs
        #
        ('RDR',): RecordFormat(reclen=RDR, auxlen=AUX),
        #
        # DEC (US EDR)
        #
        ('DEC',): RecordFormat(reclen=USEDR, auxlen=None),
        #
        # RGRAM (US RDR)
        #
        ('RGRAM',): RecordFormat(reclen=USRDR, auxlen=100),
        ('UPB',): RecordFormat(reclen=USRDR, auxlen=None),
        ('FPB',): RecordFormat(reclen=USRDR, auxlen=None),
        ('FPB_SIM',): RecordFormat(reclen=USRDR, auxlen=None),
        ('QDA',): RecordFormat(reclen=QDA, auxlen=None), # ETMs have a lengtg, but what is it?
        #
        # SCS
        #
        ('SCS',): RecordFormat(reclen=USRDR, auxlen=None)
}

SUBSYSTEM_MODES = {
        'SS01': SHARADMode(presum=32, bits_per_sample=8),
        'SS02': SHARADMode(presum=28, bits_per_sample=6),
        'SS03': SHARADMode(presum=16, bits_per_sample=4),
        'SS04': SHARADMode(presum=8, bits_per_sample=8),
        'SS05': SHARADMode(presum=4, bits_per_sample=6),
        'SS06': SHARADMode(presum=2, bits_per_sample=4),
        'SS07': SHARADMode(presum=1, bits_per_sample=8),
        'SS08': SHARADMode(presum=32, bits_per_sample=6),
        'SS09': SHARADMode(presum=28, bits_per_sample=4),
        'SS10': SHARADMode(presum=16, bits_per_sample=8),
        'SS11': SHARADMode(presum=8, bits_per_sample=6),
        'SS12': SHARADMode(presum=4, bits_per_sample=4),
        'SS13': SHARADMode(presum=2, bits_per_sample=8),
        'SS14': SHARADMode(presum=1, bits_per_sample=6),
        'SS15': SHARADMode(presum=32, bits_per_sample=4),
        'SS16': SHARADMode(presum=28, bits_per_sample=8),
        'SS17': SHARADMode(presum=16, bits_per_sample=6),
        'SS18': SHARADMode(presum=8, bits_per_sample=4),
        'SS19': SHARADMode(presum=4, bits_per_sample=8),
        'SS20': SHARADMode(presum=2, bits_per_sample=6),
        'SS21': SHARADMode(presum=1, bits_per_sample=4),
}
