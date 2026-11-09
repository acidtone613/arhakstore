"""ShardRing: one shared pool of RS shard files for many blobs.

The point of this layout: 48 shard files per *entry vault* vs O(n) files for
an entire archive. Same GF(2^8) codec, shared file pool, per-column digests,
whole-position loss simulation, and dedup by content.
"""

import random

import pytest

from arhakstore.store.ring import ShardRing, RingError


@pytest.fixture
def payloads():
    rng = random.Random(7)
    def make(n):
        return bytes(rng.randrange(256) for _ in range(n))
    return [make(300), make(5 * 48 + 7), make(1000)]


@pytest.fixture
def ring(tmp_path):
    return ShardRing(tmp_path / "ring")


def test_add_read_roundtrip(ring, payloads):
    ids = [ring.add_bytes(d, name=str(i)) for i, d in enumerate(payloads)]
    assert ids == [0, 1, 2]
    for i, d in enumerate(payloads):
        assert ring.read_bytes(i) == d
    # O(n) files on disk, not 48 per blob
    import os
    files = [f for f in (ring.shard_dir).glob("ring_shard_*.bin")]
    assert len(files) == ring.n
    st = ring.status()
    assert st["blobs"] == 3
    assert st["columns_ok"] == st["columns_total"]


def test_empty_blob_roundtrip(ring):
    assert ring.read_bytes(ring.add_bytes(b"", name="empty")) == b""


def test_dedupe_by_content(ring, payloads):
    i0 = ring.add_bytes(payloads[0])
    assert i0 == 0
    assert ring.add_bytes(payloads[0]) == i0  # deduped, same id
    assert ring.read_bytes(i0) == payloads[0]


def test_whole_position_death_and_heal(ring, payloads):
    for i, d in enumerate(payloads):
        ring.add_bytes(d, name=str(i))
    ring.kill_positions([0, 1, 2])
    st = ring.status()
    assert st["dead_positions"] == [0, 1, 2]
    for row in st["blob_status"]:
        assert row["missing"] == 3 and row["healable"]
    for i, d in enumerate(payloads):
        assert ring.read_bytes(i) == d  # still readable from the other 45
    healed = ring.heal()
    assert len(healed) == 3 * 3
    assert ring.status()["dead_positions"] == []
    for i, d in enumerate(payloads):
        assert ring.read_bytes(i) == d
    st = ring.status()
    assert st["columns_ok"] == st["columns_total"]


def test_column_corruption_only_hurts_its_blob(ring):
    d0 = bytes(random.Random(1).randrange(256) for _ in range(300))
    d1 = bytes(random.Random(2).randrange(256) for _ in range(300))
    i0, i1 = ring.add_bytes(d0, name="a"), ring.add_bytes(d1, name="b")
    # corrupt one symbol inside blob 0's column at ring position 5
    body = bytearray(ring._pos_file(5).read_bytes())
    body[3 + 2] ^= 0xFF  # second byte of blob 0's segment (position 5)
    ring._pos_file(5).write_bytes(bytes(body))
    reviewed, corrupt, missing = ring._verified_columns()
    assert 5 in corrupt[0]
    assert 5 not in corrupt[1] and 5 not in missing[0]
    assert ring.read_bytes(i0) == d0  # rebuilt from the other columns
    healed = ring.heal()
    assert (0, 5) in healed or (1, 5) in healed
    _, corrupt, _ = ring._verified_columns()
    assert 5 not in corrupt[0] and not corrupt[0]
    assert ring.read_bytes(i0) == d0


def test_delete_shrinks_and_renumbers(ring):
    d0 = bytes(random.Random(3).randrange(256) for _ in range(300))
    d1 = bytes(random.Random(4).randrange(256) for _ in range(900))
    d2 = bytes(random.Random(5).randrange(256) for _ in range(500))
    ring.add_bytes(d0, name="a")
    ring.add_bytes(d1, name="b")
    ring.add_bytes(d2, name="c")
    ring.delete(1)
    st = ring.status()
    assert st["blobs"] == 2
    assert ring.read_bytes(0) == d0
    assert ring.read_bytes(1) == d2
    # every position file is consistent and shorter
    for j in range(ring.n):
        path = ring._pos_file(j)
        if path.exists():
            assert path.read_bytes()[:3] == ring._golay.encode_24(j).to_bytes(3, "big")


def test_reopen_after_close(ring, payloads):
    ids = [ring.add_bytes(d) for d in payloads]
    ring2 = ShardRing(ring.root)  # fresh object reads the manifest
    assert ring2.status()["blobs"] == 3
    for i, d in enumerate(payloads):
        assert ring2.read_bytes(i) == d
    ring2.kill_positions([7])
    ring3 = ShardRing(ring.root)
    assert ring3.status()["dead_positions"] == [7]


def test_unknown_blob_raises(ring):
    with pytest.raises(RingError):
        ring.read_bytes(99)
    with pytest.raises(RingError):
        ring.delete(0)


def test_empty_ring_status_kill_heal_noop(tmp_path):
    """A freshly-created ring with no blobs must never crash on status/damage."""
    r = ShardRing(tmp_path / "fresh")
    st = r.status()
    assert st["blobs"] == 0 and st["dead_positions"] == []
    assert r.kill_positions([3, 7]) == [3, 7]  # recorded as dead, nothing to heal
    assert r.heal() == []                      # heal restores the ring to clean
    assert r.status()["dead_positions"] == []
    r2 = ShardRing(tmp_path / "fresh")  # reopen: empty ring stays clean
    assert r2.status()["blobs"] == 0