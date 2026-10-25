"""Generate the bundled sample datasets into ./data."""
from pathlib import Path

from omniplan.data_gen import write

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data"
    write(out)
    print(f"Wrote sample data to {out}")
