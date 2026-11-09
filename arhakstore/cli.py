"""Command-line interface for the associative (merged) half of arhakstore.

    python -m arhakstore put --archive A --key k.npy --value v.npy
    python -m arhakstore put-bytes --archive A --key k.npy --file payload.bin
    python -m arhakstore get --archive A 3 [-o out.bin]
    python -m arhakstore lookup --archive A --query q.npy [--top 5]
    python -m arhakstore kill --archive A 2 9 15
    python -m arhakstore heal --archive A
    python -m arhakstore verify --archive A
    python -m arhakstore stats --archive A
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .bridge.archive import AssociativeArchive
from .store.vault import VaultError


def _load_vec(path) -> np.ndarray:
    p = str(path)
    if p.endswith(".npy"):
        return np.load(p)
    return np.loadtxt(p, dtype=float)


def _jsum(name: str, value) -> None:
    print(json.dumps({name: value}, indent=2, default=float))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="arhakstore", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_put = sub.add_parser("put", help="store a float-vector entry")
    p_put.add_argument("--archive", required=True)
    p_put.add_argument("--key", required=True)
    p_put.add_argument("--value", required=True)
    p_put.add_argument("--name", default=None)

    p_pb = sub.add_parser("put-bytes", help="store an arbitrary-bytes entry")
    p_pb.add_argument("--archive", required=True)
    p_pb.add_argument("--key", required=True)
    p_pb.add_argument("--file", required=True)
    p_pb.add_argument("--name", default=None)

    p_get = sub.add_parser("get", help="read the exact payload of an entry")
    p_get.add_argument("--archive", required=True)
    p_get.add_argument("id", type=int)
    p_get.add_argument("-o", "--out", default=None)
    p_get.add_argument("--no-heal", action="store_true")

    p_lk = sub.add_parser("lookup", help="Steiner-attention lookup")
    p_lk.add_argument("--archive", required=True)
    p_lk.add_argument("--query", required=True)
    p_lk.add_argument("--top", type=int, default=5)
    p_lk.add_argument("--tick", type=int, default=None)
    p_lk.add_argument("--no-gate", action="store_true")

    p_kill = sub.add_parser("kill", help="kill atoms across all entries")
    p_kill.add_argument("--archive", required=True)
    p_kill.add_argument("atoms", nargs="+", type=int)

    p_heal = sub.add_parser("heal", help="heal semantic + archival planes")
    p_heal.add_argument("--archive", required=True)

    p_ver = sub.add_parser("verify", help="verify every entry vault")
    p_ver.add_argument("--archive", required=True)

    p_st = sub.add_parser("stats", help="show memory-layer + archive stats")
    p_st.add_argument("--archive", required=True)

    args = parser.parse_args(argv)

    try:
        if args.cmd == "put":
            a = AssociativeArchive(args.archive)
            eid = a.put(_load_vec(args.key), _load_vec(args.value), meta=args.name)
            a.save()
            _jsum("entry_id", eid)
        elif args.cmd == "put-bytes":
            a = AssociativeArchive(args.archive)
            eid = a.put_bytes(_load_vec(args.key), Path(args.file).read_bytes(),
                              meta=args.name)
            a.save()
            _jsum("entry_id", eid)
        elif args.cmd == "get":
            a = AssociativeArchive(args.archive)
            payload = a.get(args.id, auto_heal=not args.no_heal)
            if args.out:
                if isinstance(payload, np.ndarray):
                    payload = payload.astype(np.float32).tobytes()
                Path(args.out).write_bytes(payload)
                _jsum("out", args.out)
            else:
                if isinstance(payload, np.ndarray):
                    np.save(sys.stdout.buffer, payload)

        elif args.cmd == "lookup":
            a = AssociativeArchive(args.archive)
            r = a.lookup(_load_vec(args.query), tick=args.tick,
                         top=args.top, gate=not args.no_gate)
            _jsum("lookup", {
                "top": r["top"],
                "hexad": r["hexad"],
                "aggregate": r["aggregate"].tolist(),
            })
        elif args.cmd == "kill":
            a = AssociativeArchive(args.archive)
            killed = a.kill_atoms(args.atoms)
            _jsum("killed", killed)
        elif args.cmd == "heal":
            a = AssociativeArchive(args.archive)
            res = a.heal()
            a.save()
            _jsum("heal", res)
        elif args.cmd == "verify":
            a = AssociativeArchive(args.archive)
            _jsum("verify", a.verify())
        elif args.cmd == "stats":
            a = AssociativeArchive(args.archive)
            _jsum("stats", {
                "entries": a.n_entries,
                "mem": a.mem.stats(),
            })
        else:
            parser.error("unknown command")
    except (VaultError, OSError, ValueError) as exc:
        print(f"arhakstore: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())