""" Project initialization and common objects. """
import os
from pathlib import Path

workdir = Path(os.getenv('WORKDIR', '.'))
cachedir = workdir / 'cache'
