"""Generate a small fictional inventory problem without reading external data."""
import csv
from pathlib import Path
import random


def generate(destination=None):
    destination = Path(destination or Path(__file__).parent / "examples")
    destination.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    items = [f"SYN-ITEM-{i:02}" for i in range(1, 5)]
    with (destination / "demand.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["week", *items])
        for week in range(1, 53):
            writer.writerow([week, *(rng.randint(3, 8) for _ in items)])
    with (destination / "limits.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["item_id", "capacity", "initial_stock", "replenishment"])
        for item in items:
            writer.writerow([item, 60, 30, 25])


if __name__ == "__main__":
    generate()
