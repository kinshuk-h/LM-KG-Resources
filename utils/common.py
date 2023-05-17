"""

    common
    ~~~~~~

    Submodule for common utilities for use across the package.
    Provides functions for formatting and preprocessing data.

    Author: Kinshuk Vasisht
    Version: 1.0

"""

import re
import unicodedata

WHITESPACE_CHAR_REGEX = re.compile(r"(?ui)\s+")
SPECIAL_CHAR_REGEX    = re.compile(r"(?ui)(?:[^\w\s]\s*)*[^\w\s]")

def preprocess(entity):
    """ Basic preprocessing: removes unwanted special characters, extra spaces and normalizes to NFKC. """
    entity = SPECIAL_CHAR_REGEX.sub("", entity).strip()
    entity = WHITESPACE_CHAR_REGEX.sub(" ", entity).strip()
    return unicodedata.normalize('NFKC', entity)

def pathsafe(filename):
    """ Returns a santized, path-safe version of a filename. """
    return re.sub(r'[:/\\|*]', '-', re.sub(r'[?\"<>]', '', filename))

__TIME_UNITS__    = [ 'ns', 'us', 'ms', 's', 'm', 'h', 'd', 'w', 'y' ]
__TIME_FACTORS__  = [ 1000, 1000, 1000, 60, 60, 24, 7, 52 ]

def format_duration(time_in_secs: float) -> str:
    """ Formats a given time duration in seconds to a human-readable format.

    Args:
        time_in_secs (float): Time duration to format, in seconds.
            Can have fractional component for subsecond time.

    Returns:
        str: Formatted duration string.
    """
    time_units, index = [], 0
    time_in_secs = int(time_in_secs * 1e9)

    while time_in_secs > 0 and index < len(__TIME_FACTORS__):
        unit_time = time_in_secs % __TIME_FACTORS__[index]
        if unit_time != 0:
            time_units.append(f"{unit_time}{__TIME_UNITS__[index]}")
        time_in_secs //= __TIME_FACTORS__[index]
        index += 1

    if time_in_secs > 0:
        time_units.append(f"{time_in_secs}{__TIME_UNITS__[-1]}")

    return ' '.join(reversed(time_units)) or "0s"