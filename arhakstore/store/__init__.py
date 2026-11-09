"""arhakstore.store — dmfstore: self-healing redundant store on the DMF RS tower.

Profiles (from catalog/README.md "Reed-Solomon Hierarchy"):
    engine     RS(253, 221, 33)   GF(2^8)   default, survives 32/253 loss
    quadallic  RS(48, 12, 37)     GF(2^8)   deep archive, survives 36/48
    binary     RS(23, 12, 12)     GF(2^8)   toy/demo
"""

from .golay24 import GolayCode
from .gf256 import GF256
from .rs import ReedSolomon, PROFILES, DecodeError
from .shards import Manifest, b2b
from .vault import Vault, VaultError, DEFAULT_PROFILE
from .ring import ShardRing, RingError

__version__ = "0.2.0"

__all__ = [
    "GF256",
    "ReedSolomon",
    "GolayCode",
    "Vault",
    "Manifest",
    "VaultError",
    "ShardRing",
    "RingError",
    "DecodeError",
    "PROFILES",
    "DEFAULT_PROFILE",
    "b2b",
    "__version__",
]