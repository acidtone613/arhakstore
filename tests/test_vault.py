"""End-to-end vault tests: archive -> damage -> heal -> extract."""

import os
import random
from pathlib import Path

import pytest

from arhakstore.store.vault import Vault, VaultError


@pytest.fixture
def vault_dir(tmp_path):
    d = tmp_path / "vault"
    return d


def _make_payload(path, n_bytes, rng):
    data = bytes(rng.randrange(256) for _ in range(n_bytes))
    path.write_bytes(data)
    return data


def test_roundtrip_binary_profile(tmp_path):
    src = tmp_path / "hello.bin"
    data = _make_payload(src, 1200, random.Random(1))
    v = Vault(tmp_path / "v1")
    v.archive(src, profile="binary")  # n=23 k=12
    st = v.status()
    assert st["ok"] == 23 and st["missing"] == 0
    out = tmp_path / "out.bin"
    v.extract(out)
    assert out.read_bytes() == data


def test_roundtrip_engine_profile(tmp_path):
    src = tmp_path / "engine.bin"
    data = _make_payload(src, 5 * 221 + 19, random.Random(2))
    v = Vault(tmp_path / "v2")
    v.archive(src, profile="engine")
    st = v.status()
    assert st["n"] == 253 and st["k"] == 221
    assert st["ok"] == 253 and st["redundant"] == 32
    out = tmp_path / "out2.bin"
    v.extract(out)
    assert out.read_bytes() == data


@pytest.mark.parametrize("n_remove", [1, 8, 32])
def test_extract_survives_loss(tmp_path, n_remove):
    src = tmp_path / "loss.bin"
    data = _make_payload(src, 4000, random.Random(3))
    v = Vault(tmp_path / f"v3_{n_remove}")
    v.archive(src, profile="engine")
    for j in range(n_remove):
        (tmp_path / f"v3_{n_remove}" / "shards" / f"shard_{j:03d}.bin").unlink()
    st = v.status()
    assert st["missing"] == n_remove
    assert st["healable"]
    out = tmp_path / f"out3_{n_remove}.bin"
    v.extract(out)
    assert out.read_bytes() == data


def test_heal_regenerates_exactly(tmp_path):
    src = tmp_path / "heal.bin"
    data = _make_payload(src, 300 * 221, random.Random(4))
    v = Vault(tmp_path / "v4")
    v.archive(src, profile="engine")
    # wipe out shards 0..20 (21 lost) — systematic shards => forces decode path
    for j in range(21):
        v._shard_file(j).unlink()
    verified_before = len(v.verified_positions())
    assert verified_before == 253 - 21
    healed = v.heal()
    assert set(healed) == set(range(21))
    assert v.status()["missing"] == 0
    assert len(v.verified_positions()) == 253  # digests match the manifest
    out = tmp_path / "out4.bin"
    v.extract(out)
    assert out.read_bytes() == data


def test_corrupt_shard_detected_and_healed(tmp_path):
    src = tmp_path / "corrupt.bin"
    data = _make_payload(src, 900, random.Random(5))
    v = Vault(tmp_path / "v5")
    v.archive(src, profile="quadallic")  # n=48 k=12
    # silently corrupt one shard body
    f = v._shard_file(9)
    blob = bytearray(f.read_bytes())
    blob[3 + 7] ^= 0xFF
    blob[3 + 11] ^= 0x01
    f.write_bytes(bytes(blob))
    st = v.status()
    assert st["corrupt"] == 1
    # data is still recoverable from the other k shards -> extract succeeds
    out = tmp_path / "out5.bin"
    v.extract(out)
    assert out.read_bytes() == data
    # heal demotes the corrupt shard to "lost" and regenerates it exactly
    healed = v.heal()
    assert healed == [9]
    assert v.status()["corrupt"] == 0
    assert len(v.verified_positions()) == 48
    out2 = tmp_path / "out5b.bin"
    v.extract(out2)
    assert out2.read_bytes() == data


def test_unhealable_without_quorum(tmp_path):
    src = tmp_path / "q.bin"
    data = _make_payload(src, 800, random.Random(9))
    v = Vault(tmp_path / "vq")
    v.archive(src, profile="binary")  # k=12 of n=23
    for j in range(13):
        v._shard_file(j).unlink()  # leave only 10 < k
    assert not v.status()["healable"]
    with pytest.raises(VaultError):
        v.extract(tmp_path / "nope.bin")
    with pytest.raises(VaultError):
        v.heal()


def test_header_protection(tmp_path):
    src = tmp_path / "hdr.bin"
    data = _make_payload(src, 500, random.Random(6))
    v = Vault(tmp_path / "v6")
    v.archive(src, profile="binary")
    # flip 3 bits in the header of shard 5 — should still self-identify
    f = v._shard_file(5)
    blob = bytearray(f.read_bytes())
    blob[0] ^= 0x80
    blob[1] ^= 0x40
    blob[2] ^= 0x20
    f.write_bytes(bytes(blob))
    assert v.verified_positions().get(5) is not None
    assert v.status()["ok"] == 23


def test_manifest_resilience_and_size_correct(tmp_path):
    src = tmp_path / "m.bin"
    data = _make_payload(src, 12345, random.Random(7))
    v = Vault(tmp_path / "v7")
    v.archive(src, profile="engine")
    m, rs = v.load()
    assert m.size == 12345
    assert m.chunks == (12345 + 221 - 1) // 221
    # shard stripe lengths
    assert len(v.verified_positions()[42]) == m.chunks


def test_archive_bytes_and_extract_bytes(tmp_path):
    data = bytes(random.Random(11).randrange(256) for _ in range(600))
    v = Vault(tmp_path / "v9")
    v.archive_bytes(data, name="mem.bin", profile="quadallic")
    assert v.extract_bytes() == data
    assert v.status()["ok"] == 48


def test_archive_empty_file(tmp_path):
    src = tmp_path / "empty.bin"
    src.write_bytes(b"")
    v = Vault(tmp_path / "v8")
    v.archive(src, profile="binary")
    out = tmp_path / "out8.bin"
    v.extract(out)
    assert out.read_bytes() == b""


# ---------------------------------------------------------------------------
# Errors + erasures decoding (Berlekamp-Massey + Forney) and full verify
# ---------------------------------------------------------------------------

def test_decode_bytes_recovers_undetected_errors(tmp_path):
    """Symbol drift that is NOT digest-flagged is still corrected: any
    2*nu + mu <= r pattern per chunk is decoded exactly."""
    rng = random.Random(41)
    data = bytes(rng.randrange(256) for _ in range(300))
    v = Vault(tmp_path / "decode")
    v.archive_bytes(data, profile="quadallic")  # n=48 k=12, r=36, t=18
    # corrupt a handful of bytes spread across three shard bodies (<=18/chunk)
    for j in [5, 12, 30]:
        f = v._shard_file(j)
        blob = bytearray(f.read_bytes())
        for off in [3, 10, 22]:
            blob[off] ^= 0xA5
        f.write_bytes(bytes(blob))
    st = v.status()
    assert st["corrupt"] == 3
    found = v.decode_bytes(digest_demotion=False)  # don't drop the dirty shards
    assert found == data


def test_decode_bytes_with_digest_demotion(tmp_path):
    rng = random.Random(43)
    data = bytes(rng.randrange(256) for _ in range(600))
    v = Vault(tmp_path / "decode2")
    v.archive_bytes(data, profile="engine")  # 600 bytes -> 3 chunks
    blob = bytearray(v._shard_file(7).read_bytes())
    blob[3 + 1] ^= 0x11  # flip one symbol in chunk 1 of shard 7 (<= t=16)
    v._shard_file(7).write_bytes(bytes(blob))
    # both paths return the exact data
    assert v.decode_bytes(digest_demotion=True) == data
    assert v.decode_bytes(digest_demotion=False) == data


def test_decode_bytes_beyond_budget_raises(tmp_path):
    rng = random.Random(44)
    data = bytes(rng.randrange(256) for _ in range(120))
    v = Vault(tmp_path / "decode3")
    v.archive_bytes(data, profile="binary")  # t=5 per chunk
    for j in range(6):  # 6 errors in ONE chunk (chunk 3) across 6 shards
        blob = bytearray(v._shard_file(j).read_bytes())
        blob[3 + 3] ^= 0xFF
        v._shard_file(j).write_bytes(bytes(blob))
    with pytest.raises(VaultError):
        v.decode_bytes(digest_demotion=False)


def test_verify_full_rehashes(tmp_path):
    rng = random.Random(45)
    data = bytes(rng.randrange(256) for _ in range(900))
    v = Vault(tmp_path / "vfull")
    v.archive_bytes(data, profile="quadallic")
    r = v.verify()
    assert r["file_ok"] is True and r["full"] is False
    r = v.verify(full=True)
    assert r["file_ok"] is True and r["full"] is True
    # after healing lost shards, full verify still passes
    for j in [0, 1, 2]:
        v._shard_file(j).unlink()
    v.heal()
    assert v.verify(full=True)["file_ok"]


def test_verify_full_holds_with_healable_corruption(tmp_path):
    rng = random.Random(46)
    data = bytes(rng.randrange(256) for _ in range(300))
    v = Vault(tmp_path / "vfullbad")
    v.archive_bytes(data, profile="quadallic")  # r=36 budget
    for j in range(20):  # 20 digest-flagged corruptions <= budget
        blob = bytearray(v._shard_file(j).read_bytes())
        blob[3 + 1] ^= 0xFF
        v._shard_file(j).write_bytes(bytes(blob))
    assert v.verify(full=True)["file_ok"] is True


def test_verify_full_reports_failure_beyond_budget(tmp_path):
    rng = random.Random(47)
    data = bytes(rng.randrange(256) for _ in range(200))
    v = Vault(tmp_path / "vfulldead")
    v.archive_bytes(data, profile="binary")  # r=11 budget
    for j in range(12):  # 12 erasures > 11
        v._shard_file(j).unlink()
    r = v.verify(full=True)
    assert r["file_ok"] is False