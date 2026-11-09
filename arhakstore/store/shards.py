"""Chunking, content addressing, and the vault manifest.

A vault stores one archived blob as n shard files. The blob is split into
chunks of k bytes; chunk c is RS-encoded into n symbols; shard j holds
symbol j of every chunk (a horizontal stripe), plus a Golay-protected header
that self-identifies the shard index.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from .golay24 import GolayCode, K as GOLAY_MSG_BITS

MANIFEST_VERSION = 1


def b2b(data: bytes, size: int = 32) -> str:
    return hashlib.blake2b(data, digest_size=size).hexdigest()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Manifest:
    name: str
    profile: str
    n: int
    k: int
    size: int
    pad: int
    chunks: int
    file_digest: str
    shard_digests: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": MANIFEST_VERSION,
                "name": self.name,
                "profile": self.profile,
                "n": self.n,
                "k": self.k,
                "size": self.size,
                "pad": self.pad,
                "chunks": self.chunks,
                "file_digest": self.file_digest,
                "shard_digests": {str(j): d for j, d in self.shard_digests.items()},
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        data = json.loads(text)
        return cls(
            name=data["name"],
            profile=data["profile"],
            n=data["n"],
            k=data["k"],
            size=data["size"],
            pad=data["pad"],
            chunks=data["chunks"],
            file_digest=data["file_digest"],
            shard_digests={int(j): d for j, d in data["shard_digests"].items()},
        )


# --------------------------------------------------------------------------- #
# Shard file layout:
#   bytes 0..2   Golay-protected shard index (12-bit message -> 24-bit unit)
#   bytes 3..    one symbol (byte) per chunk, chunk index = offset - 3
# --------------------------------------------------------------------------- #
def shard_header(index: int, golay: GolayCode) -> bytes:
    unit = golay.encode_24(index & 0xFFF)
    return unit.to_bytes(3, "big")


def parse_shard_header(header: bytes, golay: GolayCode) -> int:
    """Return the protected shard index. Raises ValueError if uncorrectable."""
    if len(header) != 3:
        raise ValueError(f"bad shard header length {len(header)}")
    msg, _ = golay.decode_24(int.from_bytes(header, "big"))
    return msg