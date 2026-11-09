"""`python -m arhakstore ...` entry point (associative CLI)."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())