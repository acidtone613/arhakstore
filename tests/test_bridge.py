"""The merge's core claim: AR-HAK semantics over EXACT erasure healing.

kill_atoms deletes the 48 shard files (the atoms) of every entry vault;
heal regenerates the dead bytes exactly, so get(entry_id) is bit-identical
to the original -- unlike the semantic layer, which heals approximately.
"""

import numpy as np
import pytest

from arhakstore.bridge.archive import AssociativeArchive, pack_value, unpack_value
from arhakstore.store.vault import Vault, VaultError

N_ENTRIES = 12
N_ATOMS = 48


@pytest.fixture
def archive(tmp_path):
    a = AssociativeArchive(tmp_path / "archive", seed=7)
    rng = np.random.default_rng(7)
    keys = []
    for i in range(N_ENTRIES):
        k = np.zeros(48)
        k[i % 48] = 1.0
        k += 0.1 * rng.standard_normal(48)
        v = rng.standard_normal(96)
        a.put(k, v, meta=f"mem-{i:03d}")
        keys.append(k)
    return a, keys


def test_put_get_exact_roundtrip(archive):
    a, keys = archive
    for i in range(N_ENTRIES):
        orig = a.mem.values[i]
        got = a.get(i)
        np.testing.assert_array_equal(got, orig.astype(np.float32).astype(np.float64))


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(1)
    v = rng.standard_normal(96)
    np.testing.assert_allclose(unpack_value(pack_value(v)), v.astype(np.float32),
                               rtol=0, atol=0)


def test_kill_atoms_then_get_is_exact(archive):
    a, keys = archive
    originals = [a.get(i) for i in range(N_ENTRIES)]
    killed = a.kill_atoms([2, 9, 15, 27, 40])
    assert len(killed) == 5
    # every vault now reports the killed shards missing
    for i in range(N_ENTRIES):
        st = a._vault(i).status()
        assert st["missing"] == 5
        assert st["ok"] == N_ATOMS - 5
    # read-through: get() heals on demand and returns the EXACT original
    for i in range(N_ENTRIES):
        np.testing.assert_array_equal(a.get(i), originals[i])


def test_heal_regenerates_exactly(archive):
    a, keys = archive
    originals = [a.get(i) for i in range(N_ENTRIES)]
    a.kill_atoms([0, 3, 6, 9, 12, 15, 18])
    res = a.heal()
    assert res["verified"]["all_clean"]
    assert res["verified"]["shards_ok"] == N_ENTRIES * N_ATOMS
    assert res["entries_healed"] == N_ENTRIES
    for i in range(N_ENTRIES):
        np.testing.assert_array_equal(a.get(i), originals[i])


def test_lookup_routes_to_the_right_entry(archive):
    a, keys = archive
    q = keys[3] + 0.1 * np.random.default_rng(3).standard_normal(48)
    r = a.lookup(q, top=3)
    assert r["top"][0][0] == 3  # nearest entry wins
    assert r["weights"][3] > 0


def test_lookup_after_heal_routes_again(archive):
    a, keys = archive
    a.kill_atoms([2, 9, 15, 27, 40])
    a.heal()
    q = keys[3] + 0.1 * np.random.default_rng(3).standard_normal(48)
    r = a.lookup(q, top=3)
    assert r["top"][0][0] == 3


def test_persistence_roundtrip(archive, tmp_path):
    a, keys = archive
    a.save()
    b = AssociativeArchive(tmp_path / "archive")
    b.load()
    assert b.n_entries == N_ENTRIES
    for i in range(N_ENTRIES):
        np.testing.assert_array_equal(a.get(i), b.get(i))


def test_get_rejects_unhealable_without_auto(tmp_path):
    a = AssociativeArchive(tmp_path / "a", seed=1)
    rng = np.random.default_rng(1)
    a.put(np.zeros(48), rng.standard_normal(96))
    a.kill_atoms(list(range(40)))  # only 8 < k=12 survivors
    with pytest.raises(VaultError):
        a.get(0, auto_heal=False)


def test_lookup_empty_archive_raises(tmp_path):
    a = AssociativeArchive(tmp_path / "empty")
    with pytest.raises(VaultError):
        a.lookup(np.zeros(48))


def test_put_bytes_roundtrip_bit_exact(tmp_path):
    a = AssociativeArchive(tmp_path / "bytes", seed=1)
    payload = bytes(range(256)) * 8  # arbitrary binary, 2048 bytes
    key = np.zeros(48)
    eid = a.put_bytes(key, payload, meta="raw")
    assert a.get_bytes(eid) == payload  # raw, exact
    assert a.get(eid) == payload        # get() routes to the bytes path
    assert a.kinds == ["bytes"]
    # persistence keeps it exact
    a.save()
    b = AssociativeArchive(tmp_path / "bytes")
    b.load()
    assert b.get_bytes(eid) == payload


def test_put_bytes_heals_exactly(tmp_path):
    a = AssociativeArchive(tmp_path / "bytes2", seed=1)
    payload = b"exact-archive-bytes-" * 40
    eid = a.put_bytes(np.zeros(48), payload)
    a.kill_atoms([4, 11, 33])
    assert a.get_bytes(eid) == payload  # read-through heal, bit-exact
    a.heal()
    assert a.get_bytes(eid) == payload


def test_mixed_kinds_roundtrip(tmp_path):
    a = AssociativeArchive(tmp_path / "mixed", seed=1)
    rng = np.random.default_rng(2)
    k1 = rng.standard_normal(48)
    e0 = a.put(np.zeros(48), rng.standard_normal(96))
    e1 = a.put_bytes(np.ones(48), b"\x00\x01\x02")
    assert a.kinds == ["ndarray", "bytes"]
    assert isinstance(a.get(e0), np.ndarray)
    assert a.get(e1) == b"\x00\x01\x02"
    a.save()
    b = AssociativeArchive(tmp_path / "mixed")
    b.load()
    assert b.kinds == ["ndarray", "bytes"]