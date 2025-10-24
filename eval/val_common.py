# eval/val_common.py
"""
Compatibility shim: re-export evaluation utilities from gcs_core.val_common
to avoid code duplication.
"""
from gcs_core.val_common import *  # noqa: F401,F403
