"""PDDP-only evaluation entry; legacy PDP outputs are rejected."""
import sys
from data_generation.scripts.pdd import main

if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])
