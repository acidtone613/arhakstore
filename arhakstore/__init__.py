"""arhakstore — AR-HAK associative memory over dmfstore's exact erasure store.

A merge of two sibling codebases that grew from the same research corpus:

  * `mem`   AR-HAK (Attention-Routed Holographic Associative Kernel): an LLM
            memory bank with Steiner/Witt hexad attention, a 276-slot hologram
            and 3-level self-healing over its 48-atom address plane.
  * `store` dmfstore: byte-exact self-healing storage over the corpus's
            Reed-Solomon tower (GF(2^8), engine RS(253,221,33) / quadallic
            RS(48,12,37) / binary RS(23,12,12)) with Golay-protected shards.
  * `bridge` AssociativeArchive: attention-addresses an entry, stores its
            payload as an erasure-coded atom vault, and heals dead atoms
            EXACTLY (GF erasure reconstruction) instead of approximately
            (Z7 group-mate mean).
  * `store.ring` ShardRing: many blobs sharing one pool of RS shard files
            (O(n) files per archive, not 48 per entry), with per-column
            digests, shared loss healing, and dedupe.

Neither source project is modified; this folder is a clean merge.
"""

__version__ = "0.2.0"

from .bridge.archive import AssociativeArchive
from .store import (
    GF256,
    DecodeError,
    GolayCode,
    PROFILES,
    ReedSolomon,
    RingError,
    ShardRing,
    Vault,
    VaultError,
)

__all__ = [
    "AssociativeArchive",
    "Vault",
    "VaultError",
    "ShardRing",
    "RingError",
    "DecodeError",
    "ReedSolomon",
    "GolayCode",
    "GF256",
    "PROFILES",
    "__version__",
]