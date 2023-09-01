"""A python package for simulating the sampling behaviors of enhanced sampling simulations"""

# Add imports here
# from .sampling_simulator import *

# Handle versioneer
from ._version import get_versions  # noqa: ABS101

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
