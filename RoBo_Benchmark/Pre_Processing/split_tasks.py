"""Teilt tasks.csv in N Chunks auf - einen selbststaendigen Ordner pro Instanz.

Jeder Instanz-Ordner enthaelt alles, was diese Instanz zum Rechnen braucht:
tasks.csv (nur ihr Anteil) + tensor_noise.npy (identische Kopie fuer alle).
Diese Ordner werden anschliessend auf die jeweilige LRZ-Instanz kopiert
(z.B. per scp/rsync).

    python Pre_Processing/split_tasks.py --instanzen 4   (aus der ROBO_Benchmark-Wurzel)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pfade  # noqa: F401

import argparse
import shutil

import pandas as pd

import config


def main(n_instanzen):# ✅ 
    df_tasks = pd.read_csv(config.DATA_DIR / "tasks.csv")
    tensor_path = config.DATA_DIR / "tensor_noise.npy"

    instances_dir = config.DATA_DIR / "instances"
    instances_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_instanzen):
        chunk = df_tasks[df_tasks["task_id"] % n_instanzen == i]
        inst_dir = instances_dir / f"instance_{i}"
        inst_dir.mkdir(exist_ok=True)
        chunk.to_csv(inst_dir / "tasks.csv", index=False)
        shutil.copy(tensor_path, inst_dir / "tensor_noise.npy")
        print(f"instance_{i}: {len(chunk)} Tasks -> {inst_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instanzen", type=int, default=config.N_INSTANZEN)
    args = parser.parse_args()
    main(args.instanzen)
