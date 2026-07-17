# SPDX-License-Identifier: BSD-3-Clause
from pathlib import Path
from ..common.config import get_data_path

MOLA_REGIONS = ["GLOBAL", "POLAR"]
AREOID_RESOLUTIONS = [4,16]
COUNTS_RESOLUTIONS = [128, 256, 512]
MOLA_TYPES = ["COUNTS", "AREOID", "RADIUS", "TOPOGRAPHY"]
MOLA_RESOLUTIONS = [4,16,32,64,128,256,512]
MOLA_RES_MAP = {'c': 4, 'e':16, 'f': 32, 'g': 64, 'h':128}


MOLA_METADATA = {"FILE_RECORDS": None,
                 "RECORD_BYTES": None,
                 "LINES": None,
                 "LINE_SAMPLES": None,
                 "SAMPLE_TYPE": None,
                 "SAMPLE_BITS": None,
                 "UNIT": None,
                 "OFFSET": None,
                 "SCALING_FACTOR": None,
                 "MAP_PROJECTION_TYPE": None,
                 "A_AXIS_RADIUS": None,
                 "B_AXIS_RADIUS": None,
                 "C_AXIS_RADIUS": None,
                 "FIRST_STANDARD_PARALLEL": None,
                 "SECOND_STANDARD_PARALLEL": None,
                 "POSITIVE_LONGITUDE_DIRECTION": "EAST",
                 "CENTER_LATITUDE": None,
                 "CENTER_LONGITUDE": None,
                 "REFERENCE_LATITUDE": None,
                 "REFERENCE_LONGITUDE": None,
                 "LINE_FIRST_PIXEL": None,
                 "LINE_LAST_PIXEL": None,
                 "SAMPLE_FIRST_PIXEL": None,
                 "SAMPLE_LAST_PIXEL": None,
                 "MAP_PROJECTION_ROTATION": None,
                 "MAP_RESOLUTION": None,
                 "MAP_SCALE": None,
                 "MAXIMUM_LATITUDE": None,
                 "MINIMUM_LATITUDE": None,
                 "WESTERNMOST_LONGITUDE": None,
                 "EASTERNMOST_LONGITUDE": None,
                 "LINE_PROJECTION_OFFSET": None,
                 "SAMPLE_PROJECTION_OFFSET": None,
                 "COORDINATE_SYSTEM_TYPE": None,
                 "COORDINATE_SYSTEM_NAME": None,
                 }


MOLA_VALID = {"AREOID": {"GLOBAL": {"MAP_RESOLUTIONS": [4,16],
                                    },
                         },
              "COUNTS": {"GLOBAL": {"MAP_RESOLUTIONS": [4,16,32,64,128],
                                    },
                         "POLAR": {"MAP_RESOLUTIONS": [128, 256, 512],
                                   },
                        },
              "RADIUS": {"GLOBAL": {"MAP_RESOLUTIONS": [4,16,32,64,128],
                                    },
                         "POLAR": {"MAP_RESOLUTIONS": [128,256,512],
                                   },
                         },
              "TOPOGRAPHY": {"GLOBAL": {"MAP_RESOLUTIONS": [4,16,32,64,128],
                                        },
                             "POLAR": {"MAP_RESOLUTIONS": [128,256,512]},
                             },
              }

MOLA_PPD_CODES = {"GLOBAL": {"4": "cb",
                             "16": "eb",
                             "32": "fb",
                             "64": "gb",
                             "128": "hb",
                             },
                  }

MOLA_DATA_IDS = {"AREOID": "a",
                 "COUNTS": "c",
                 "RADIUS": "r",
                 "TOPOGRAPHY": "t"}

MOLA_FILENAME_TEMPLATES = {"GLOBAL": "meg{}{}{}{}{}.img",
                           "POLAR": "meg{}_{}_{}.img"}

def get_mola_root(override: str | Path | None = None) -> Path:
    """Resolve the MOLA data root directory.

    Args:
        override: Explicit path that takes highest priority.

    Returns:
        Resolved path to the MOLA data directory.
    """
    return get_data_path("mola", override=override)