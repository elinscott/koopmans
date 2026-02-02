import ipdb
import traceback
import sys


def _pdb_exception_hook(exception_type, exception_value, exception_traceback):
    traceback.print_exception(exception_type, exception_value, exception_traceback)
    ipdb.post_mortem(exception_traceback)

def enable_pdb():
    sys.tracebacklimit = None
    sys.excepthook = _pdb_exception_hook 
