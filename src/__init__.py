""" Project initialization and common objects. """

import logging
import os
from pathlib import Path
import sys


logging.basicConfig(
    stream=sys.stdout,
    format='[%(levelname)s][%(module)s] %(message)s',
    level=os.getenv('LOGLEVEL', 'info').upper())

workdir = Path(os.getenv('WORKDIR', '.'))
cachedir = workdir / 'cache'
