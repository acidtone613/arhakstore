# arhakstore

**AR-HAK associative memory over dmfstore's exact Reed–Solomon erasure store.**

A clean merge of two sibling codebases that grew out of the same research
corpus (the 24D-DMF "truth folder"): *AR-HAK*, an LLM memory layer with
Steiner/Witt-hexad attention, and *dmfstore*, a byte-exact self-healing
storage engine built on the corpus's verified Reed–Solomon tower over GF(2^8).

The two sources are unpublished sibling projects; this repository is the only
place they are bound together, and both are merged without modification
beyond relativized imports and a bytes-level API.

## Layout

| Package | Origin | Contents |
|---|---|---|
| `arhakstore.mem`   | AR-HAK (copied, imports relativized) | `[48,12,16]` doubled-Golay code plane, Witt hexads, Steiner attention, 276-slot hologram, Z7 group-mate repair, `MemoryLayer` |
| `arhakstore.store` | dmfstore (copied + bytes API + decoder) | GF(2^8) engine, systematic Reed–Solomon (engine/quadallic/binary), **Berlekamp–Massey + Forney errors-and-erasures decoder**, Golay-24 shard headers, `Vault`, store CLI |
| `arhakstore.store.golay24_vec` | new | AR-HAK's vectorized nearest-codeword decode ported onto the store side |
| `arhakstore.store.ring` | new | `ShardRing`: one shared pool of RS shard files for an entire archive (O(n) files, not 48 per entry) with per-column digests and whole-position healing |
| `arhakstore.bridge` | new | `AssociativeArchive` — the merge (float fan-out + raw-bytes payloads) |
| `arhakstore.cli` | new | associative CLI: `python -m arhakstore put/get/lookup/kill/heal/verify/stats` |

## What the merge proves

AR-HAK heals a killed atom's *value slices* **approximately** (mean of Z7
group-mates — meaningful entropy, not the original bytes). dmfstore heals
**exactly**: any killed position of an RS codeword is regenerated over GF(2^8)
so the bytes come back identical.

`AssociativeArchive` wires both planes to the *same* 48 atoms:

```
put(key, value)  ->  entry row (AR-HAK float bank + [48,12,16] code)
                    + quadallic vault  RS(48,12,37)  (48 shard files = atoms)
lookup(query)    ->  Steiner attention over the float bank -> top entry ids
get(id)          ->  read-through: exact payload bytes (doggedly healed)
kill_atoms(a..)  ->  zero the semantic slices AND delete shard files a..
heal()           ->  AR-HAK semantic heal (approx) + exact shard regeneration
verify()         ->  every entry vault digest-clean
```

The read path is read-through-healed: as long as any `k=12` shards survive
the killed atoms, `get(id)` returns the **bit-exact** payload (lazily healing
the dead shard files along the way), and `verify()` reports
`file_ok` from a live re-decode + re-hash when run with `full=True`.

So after killing 7 atoms: `lookup` recovers (approximate, by design), while
`get(id)` returns **bit-identical** payloads — the semantic layer is
approximate by design, the archival layer is exact by construction. Both
claims stay visible and honest.

## Decoder: Berlekamp–Massey + Forney

`store/rs.py` now decodes, not just repairs:

```
decode_correct(received, erasures=None) -> corrected codeword
```

- **Errors**: located by the classical Berlekamp–Massey algorithm, magnitudes
  by Forney (`X * Ω(x) / Λ'(x)`).
- **Erasures**: listed positions are peeled off first; any mixed pattern with
  `2*nu + mu <= r = n-k` per chunk decodes exactly.
- **Beyond that**: `DecodeError` / `VaultError`, never silent miscorrection —
  the per-chunk locator/syndromes are re-checked and the whole-blob digest
  gates the result.
- `verify --full` uses the decoder to tell you whether the current file is
  "exactly the archived bytes" instead of "the digest that's on record is".
- `ReedSolomonRepairer` still rebuilds lost shards position-by-position with a
  memoized transfer matrix (engine heal ≈ 48× faster than the naive solve).

Cross-pollination in the other direction: `store.golay24_vec` uses AR-HAK's
brute-force numpy decode (all 4096 codewords) to batch-check shard headers,
with behavior identical to the scalar syndrome-table decoder.

## Run it

```
# project root (= repo root)
pip install -e .[test]          # or: pip install numpy pytest
python -m pytest tests -q       # fast suite (store + mem + bridge + vector)
python -m pytest tests -m slow  # brute-force engine-tier tests (~15 s)
python demo/self_heal_associative.py
```

CLI of the store half (`python -m arhakstore.store`):

```
... archive data.bin -o vault
... status vault | heal vault | extract vault -O restored.bin
... verify vault            # add --full for decode+rehash, or decode (--keep-corrupt)
... ring add vault/file.bin | list | read ID -o out | heal | kill 2 9 | delete ID
```

Associative CLI (`python -m arhakstore`):

```
python -m arhakstore put  --archive A --key k.npy --value v.npy [--name tag]
python -m arhakstore put-bytes --archive A --key k.npy --file payload.bin
python -m arhakstore get  --archive A 3 [-o out.bin]        # exact bytes
python -m arhakstore lookup --archive A --query q.npy --top 5
python -m arhakstore kill --archive A 2 9 15 | heal | verify | stats
```

`put` fans a float64 vector out at float32 width (the AR-HAK semantic layer's
native precision). Anything else — full float64 arrays, text, weights, models —
goes through `put_bytes`/`get_bytes`, which store the exact byte string and
return it bit-identical, and heal exactly.

## Honest limits (inherited, not hidden)

- Memory layer semantics: attention/aggregate are floating-point *approximate*
  after heavy damage; exactness is guaranteed on the archived payloads and on
  any `ShardRing` blob.
- Naming/layout: `kill_atoms` deletes the corresponding shard files and zeroes
  the semantic slices — the vault recovers exactly, the memory layer
  approximately, and both are auditable via `verify`.
- Per-entry vaults are fine to ~a few MB; the file-count argument for
  `ShardRing` (O(n) files per archive) is now in the tree, not just on the
  roadmap. `_io.py` writes manifests atomically (temp + rename + fsync).
- Decoder cap is the honest one: any `2*nu + mu <= r` per chunk decodes
  exactly; anything heavier raises `VaultError`/`DecodeError` and
  `verify --full` reports `file_ok=False` — never silent miscorrection.
- The memory layer's research provenance (RC-343/RC-353 etc.) is presented as
  executed hypotheses, consistent with the truth-folder curation.
