from .config import backend

if backend == 'plaid':
    import plaidml.keras
    plaidml.keras.install_backend()

from . import evolution
from . import evaluation
