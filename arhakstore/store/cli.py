"""Command-line interface for the erasure-store half of arhakstore.

    python -m arhakstore.store archive data.bin -o vault          # fan-out
    python -m arhakstore.store status vault                        # live picture
    python -m arhakstore.store extract vault -O restored.bin      # rebuild from any k
    python -m arhakstore.store decode vault -O restored.bin       # errors+erasures (BM+Forney)
    python -m arhakstore.store heal vault                          # regenerate lost shards
    python -m arhakstore.store verify vault [--full]               # integrity sweep
    python -m arhakstore.store ring add|list|read|heal|delete DIR  # shared-pool archive
"""

from __future__ import annotations

import argparse
import sys

from .ring import ShardRing, RingError
from .rs import PROFILES
from .vault import Vault, VaultError


def _print_status(st: dict) -> None:
    line = (
        f"{st['profile']:>10}  n={st['n']} k={st['k']}  "
        f"ok={st['ok']} corrupt={st['corrupt']} header_bad={st['header_bad']} "
        f"missing={st['missing']}  redundancy={st['redundant']:+d}  "
        f"loss_budget={st['loss_budget']}"
    )
    print(line)
    if st["healable"]:
        print(f"  -> healable: yes (surplus {st['redundant']} shards)")
    else:
        print(f"  -> healable: NO (need {st['k']} intact, have {st['ok']})")


def _print_ring_status(st: dict) -> None:
    print(
        f"ring {st['profile']}  n={st['n']} k={st['k']}  blobs={st['blobs']}  "
        f"columns_ok={st['columns_ok']}/{st['columns_total']}  "
        f"dead_positions={st['dead_positions']}"
    )
    for row in st["blob_status"]:
        print(
            f"  #{row['id']:<4} {row['name']:<24} size={row['size']:<8} "
            f"ok={row['ok']} corrupt={row['corrupt']} missing={row['missing']} "
            f"surplus={row['surplus']:+d} healable={row['healable']}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="arhakstore", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ar = sub.add_parser("archive", help="fan a file out into n shards")
    p_ar.add_argument("file")
    p_ar.add_argument("-o", "--out", required=True, help="vault directory")
    p_ar.add_argument(
        "--profile",
        default="engine",
        choices=list(PROFILES),
        help="RS profile (default engine)",
    )

    p_ex = sub.add_parser("extract", help="rebuild the blob from any k shards")
    p_ex.add_argument("vault")
    p_ex.add_argument("-O", "--out", required=True)

    p_dc = sub.add_parser("decode", help="errors+erasures (BM+Forney) rebuild")
    p_dc.add_argument("vault")
    p_dc.add_argument("-O", "--out", required=True)
    p_dc.add_argument("--keep-corrupt",
                      action="store_true",
                      help="hand undigested corrupt symbols to the decoder as errors")

    p_st = sub.add_parser("status", help="show vault health")
    p_st.add_argument("vault")

    p_v = sub.add_parser("verify", help="recompute per-shard digests")
    p_v.add_argument("vault")
    p_v.add_argument("--full", action="store_true", help="also re-decode and re-hash the blob")

    p_heal = sub.add_parser("heal", help="regenerate missing/corrupt shards")
    p_heal.add_argument("vault")

    p_r = sub.add_parser("ring", help="shared-shard-ring archive (many blobs, O(n) files)")
    rsub = p_r.add_subparsers(dest="ring_cmd", required=True)
    p_r_add = rsub.add_parser("add", help="add a file to the ring")
    p_r_add.add_argument("dir", help="ring root")
    p_r_add.add_argument("file")
    p_r_list = rsub.add_parser("list", help="ring + per-blob status")
    p_r_list.add_argument("dir")
    p_r_read = rsub.add_parser("read", help="rebuild one blob")
    p_r_read.add_argument("dir")
    p_r_read.add_argument("id", type=int)
    p_r_read.add_argument("-O", "--out", required=True)
    p_r_heal = rsub.add_parser("heal", help="regenerate dead/corrupt columns")
    p_r_heal.add_argument("dir")
    p_r_kill = rsub.add_parser("kill", help="simulate whole-position death")
    p_r_kill.add_argument("dir")
    p_r_kill.add_argument("positions", nargs="+", type=int)
    p_r_del = rsub.add_parser("delete", help="drop a blob (rewrites + shrinks files)")
    p_r_del.add_argument("dir")
    p_r_del.add_argument("id", type=int)

    args = parser.parse_args(argv)

    try:
        if args.cmd == "archive":
            v = Vault(args.out)
            created = v.archive(args.file, profile=args.profile)
            print(f"archived {args.file} -> {created}")
            _print_status(v.status())
        elif args.cmd == "extract":
            v = Vault(args.vault)
            out = v.extract(args.out)
            print(f"reconstructed -> {out}")
            _print_status(v.status())
        elif args.cmd == "decode":
            v = Vault(args.vault)
            data = v.decode_bytes(digest_demotion=not args.keep_corrupt)
            with open(args.out, "wb") as fh:
                fh.write(data)
            print(f"decoded ({len(data)} bytes, digest verified) -> {args.out}")
        elif args.cmd == "status":
            _print_status(Vault(args.vault).status())
        elif args.cmd == "verify":
            v = Vault(args.vault)
            r = v.verify(full=args.full)
            print(
                f"verify: ok={r['ok']} corrupt={r['corrupt']} "
                f"header_bad={r['header_bad']} missing={r['missing']} "
                f"file_ok={r['file_ok']}{' (full re-hash)' if r['full'] else ''}"
            )
        elif args.cmd == "heal":
            v = Vault(args.vault)
            healed = v.heal()
            print(f"healed {len(healed)} shards: {healed}")
            _print_status(v.status())
        elif args.cmd == "ring":
            if args.ring_cmd == "add":
                ring = ShardRing(args.dir)
                blob_id = ring.add(args.file)
                print(f"added {args.file} -> blob id {blob_id}")
                _print_ring_status(ring.status())
            elif args.ring_cmd == "list":
                _print_ring_status(ShardRing(args.dir).status())
            elif args.ring_cmd == "read":
                ring = ShardRing(args.dir)
                out = ring.read(args.id, args.out)
                print(f"rebuilt blob {args.id} -> {out}")
            elif args.ring_cmd == "heal":
                ring = ShardRing(args.dir)
                healed = ring.heal()
                print(f"healed {len(healed)} (blob, position) pairs")
                _print_ring_status(ring.status())
            elif args.ring_cmd == "kill":
                ring = ShardRing(args.dir)
                dead = ring.kill_positions(args.positions)
                print(f"dead positions: {dead}")
            elif args.ring_cmd == "delete":
                ring = ShardRing(args.dir)
                ring.delete(args.id)
                print(f"deleted blob {args.id}")
                _print_ring_status(ring.status())
            else:
                parser.error("unknown ring command")
        else:
            parser.error("unknown command")
    except (VaultError, RingError, OSError, ValueError) as exc:
        print(f"arhakstore: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
