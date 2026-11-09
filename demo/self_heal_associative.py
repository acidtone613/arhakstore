"""End-to-end: AR-HAK memory + dmfstore exact erasure, kill -> heal -> verify.

Shows the merge's core contrast in one run:
  * semantic layer (lookup/aggregate) heals APPROXIMATELY (Z7 group-mate mean)
  * archival layer (get by id) heals EXACTLY (RS erasure reconstruction)

Run from the project root:
    python demo/self_heal_associative.py
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arhakstore.bridge.archive import AssociativeArchive

N_ENTRIES = 40
KILL_ATOMS = [2, 9, 15, 27, 40, 7, 22]


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="arhakstore_demo_"))
    rng = np.random.default_rng(343)

    print("== 1. populate the associative archive ==")
    a = AssociativeArchive(work / "archive", seed=343)
    keys = []
    originals = {}
    for i in range(N_ENTRIES):
        k = np.zeros(48)
        k[i % 48] = 1.0
        k += 0.1 * rng.standard_normal(48)
        v = rng.standard_normal(96)
        a.put(k, v, meta=f"mem-{i:03d}")
        keys.append(k)
        originals[i] = v.astype(np.float32)
    print(f"    entries={a.n_entries}  (each = 48-atom quadallic vault, "
          f"RS(48,12,37))")
    print(f"    shards on disk      = {a.verify()['shards_total']}")

    print("== 2. hash-addressed lookup (Steiner attention) ==")
    q = keys[11] + 0.1 * np.random.default_rng(11).standard_normal(48)
    r = a.lookup(q, top=5)
    print(f"    top match -> entry {r['top'][0][0]}  (expected 11)")

    print("== 3. kill atoms (semantic slices zeroed + shard files deleted) ==")
    a.kill_atoms(KILL_ATOMS)
    st0 = a.verify()
    print(f"    dead atoms: {KILL_ATOMS}")
    print(f"    vault health after kill: {st0['shards_ok']}/{st0['shards_total']} ok")

    print("== 4. semantic lookup while damaged (APPROXIMATE by design) ==")
    before = np.sum(a.lookup(q, top=5)["weights"] * a.mem.values.T, axis=1)
    ref = originals[11].astype(np.float64)
    fid_before = float(np.abs(np.vdot(ref, before)) /
                       (np.linalg.norm(ref) * np.linalg.norm(before) + 1e-10))
    print(f"    aggregate fidelity (damaged)  = {fid_before:.4f}")

    print("== 5. archival get() while atoms are dead (read-through heal) ==")
    got_damaged = a.get(11)
    exact_while_damaged = np.array_equal(got_damaged, originals[11])
    print(f"    get(11) == original payload: {exact_while_damaged} "
          f"(bit-exact, no heal() call needed)")

    print("== 6. heal both planes ==")
    t0 = time.time()
    summary = a.heal()
    dt = time.time() - t0
    ver = summary["verified"]
    print(f"    heal took {dt:.2f}s; regenerated {summary['shards_healed_total']} "
          f"shards across {summary['entries_healed']} entries")
    print(f"    vault health after heal: {ver['shards_ok']}/{ver['shards_total']} "
          f"ok, all_clean={ver['all_clean']}")

    print("== 7. exactness re-check: every entry bit-identical ==")
    for i in range(N_ENTRIES):
        np.testing.assert_array_equal(a.get(i), originals[i])
    print(f"    {N_ENTRIES}/{N_ENTRIES} entries exact after heal")

    print("== 8. semantic fidelity after heal (approximate architecture) ==")
    after = np.sum(a.lookup(q, top=5)["weights"] * a.mem.values.T, axis=1)
    fid_after = float(np.abs(np.vdot(ref, after)) /
                      (np.linalg.norm(ref) * np.linalg.norm(after) + 1e-10))
    print(f"    aggregate fidelity (damaged)  = {fid_before:.4f}")
    print(f"    aggregate fidelity (healed)   = {fid_after:.4f}")

    print("== 9. put_bytes: arbitrary-binary payloads, exact & healing ==")
    payload = bytes(range(256)) * 16          # 4096 raw bytes (float64, text, weights...)
    bid = a.put_bytes(np.ones(48), payload, meta="raw")
    a.kill_atoms([3, 17, 41])
    exact = a.get_bytes(bid)
    print(f"    put_bytes(4096 B), killed 3 atoms, get_bytes == exact: "
          f"{exact == payload}")

    print("== 10. undetected byte drift corrected by BM+Forney decode ==")
    v0 = a._vault(0)
    blob = bytearray(v0._shard_file(12).read_bytes())
    blob[3 + 5] ^= 0xA5; blob[3 + 17] ^= 0xA5
    v0._shard_file(12).write_bytes(bytes(blob))
    recovered = v0.decode_bytes(digest_demotion=False)
    print(f"    flipped 2 symbols in shard 12 (not digest-flagged as erasure); "
          f"decode == original: {recovered == originals[0].tobytes()}")
    print(f"    verify --full on this vault: "
          f"file_ok={v0.verify(full=True)['file_ok']}")

    print("== 11. beyond-budget corruption fails --full (honest cap) ==")
    from arhakstore.store.vault import Vault
    hard = Vault(work / "hardcap")
    hard_data = bytes(200)
    hard.archive_bytes(hard_data, profile="binary")   # r=11 budget
    for j in range(12):                                # 12 losses > 11
        hard._shard_file(j).unlink()
    print(f"    binary-tier vault with 12/23 shards gone -> "
          f"verify --full file_ok={hard.verify(full=True)['file_ok']}")

    print("== 12. shared shard ring: O(n) files for the whole archive ==")
    from arhakstore.store.ring import ShardRing
    ring = ShardRing(work / "ring")
    ring_docs = {
        0: ("README.md", Path(__file__).resolve().parent.parent.joinpath("README.md").read_bytes()),
        1: ("self_heal_associative.py", Path(__file__).resolve().read_bytes()),
        2: ("pyproject.toml", Path(__file__).resolve().parent.parent.joinpath("pyproject.toml").read_bytes()),
    }
    for _, (name, data) in ring_docs.items():
        ring.add_bytes(data, name=name)
    import os as _os
    files_on_disk = len([p for p in ring.shard_dir.glob("ring_shard_*.bin")])
    ring.kill_positions([6, 19, 30])
    healed = ring.heal()
    ok = all(ring.read_bytes(i) == ring_docs[i][1] for i in range(3))
    print(f"    3 blobs in {files_on_disk} shard files (48 each for a vault, "
          f"48 total for a ring); killed 3 positions, healed {len(healed)} "
          f"columns, all blobs bit-exact: {ok}")

    print(f"\nworking root: {work}")
    print("summary: semantic layer approximates, archival layer is exact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())