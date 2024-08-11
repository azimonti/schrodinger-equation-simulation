#!/usr/bin/env python3
'''
/************************/
/*    mod_config.py     */
/*     Version 1.0      */
/*      2024/06/02      */
/************************/
'''
import sys
from types import SimpleNamespace

palette = SimpleNamespace(
    b=(102 / 255, 204 / 255, 255 / 255),  # blue
    o=(255 / 255, 153 / 255, 102 / 255),  # orange
    r=(204 / 255, 0 / 255, 102 / 255),    # red
    g=(102 / 255, 204 / 255, 102 / 255)   # green
)

if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise 'Must be using Python 3'
    pass
