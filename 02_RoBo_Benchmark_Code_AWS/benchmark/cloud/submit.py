# cloud/submit.py
#
# Erstellt den Experimentplan genau wie main.py (create_experiment_table +
# remove_duplicate_nominal_runs) und stellt jedes Experiment als eigene Nachricht in
# die SQS-Queue. Die Worker auf den EC2-Instanzen holen sie dort ab.
# Zusaetzlich wird der aktuelle Code als Paket nach S3 hochgeladen, damit alle Worker
# genau diesen Stand (inkl. config.py) rechnen.
#
# Aufruf aus dem Ordner benchmark/:
#     python -m cloud.submit                       # kompletter Plan
#     python -m cloud.submit --max-experiments 3   # kurzer Testlauf
#     python -m cloud.submit --dry-run             # nur anzeigen, nichts senden

import argparse
import io
import json
import tarfile
from pathlib import Path

from preprocessing.experiment_design import create_experiment_table, remove_duplicate_nominal_runs
from cloud import aws_settings as aws


BENCHMARK_DIR = Path(__file__).resolve().parent.parent
# Nicht ins Code-Paket: Ergebnisse, COCO-Logs, Caches
EXCLUDE_DIRS = {"results", "exdata", "__pycache__", ".ipynb_checkpoints", ".git"}


def build_code_package():
    buffer = io.BytesIO()

    def exclude(info):
        parts = Path(info.name).parts
        return None if any(p in EXCLUDE_DIRS for p in parts) else info

    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(BENCHMARK_DIR, arcname="benchmark", filter=exclude)
    return buffer.getvalue()


def done_task_ids(s3, bucket):
    # Bereits fertige Experimente (Ergebnisdatei in S3) werden nicht erneut eingestellt
    prefix = f"{aws.RUN_NAME}/tasks/"
    done = set()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            done.add(obj["Key"][len(prefix):].removesuffix(".csv"))
    return done


def send_experiments(sqs, url, experiments):
    # send_message_batch nimmt maximal 10 Nachrichten
    for start in range(0, len(experiments), 10):
        batch = experiments[start:start + 10]
        response = sqs.send_message_batch(QueueUrl=url, Entries=[
            {"Id": str(i), "MessageBody": json.dumps(e, default=float)}
            for i, e in enumerate(batch)
        ])
        if response.get("Failed"):
            raise RuntimeError(f"Senden fehlgeschlagen: {response['Failed']}")
        print(f"  {min(start + 10, len(experiments))}/{len(experiments)} gesendet")


def main(max_experiments=None, dry_run=False, skip_code=False):
    # 1. Experimentplan erstellen - identisch zu main.py
    experiments = create_experiment_table()
    experiments = remove_duplicate_nominal_runs(experiments)
    if max_experiments is not None:
        experiments = experiments[:max_experiments]

    bucket = aws.require_bucket()
    sess = aws.session()
    s3 = sess.client("s3")

    done = done_task_ids(s3, bucket)
    experiments = [e for e in experiments if e["task_id"] not in done]
    print(f"Lauf '{aws.RUN_NAME}': {len(done)} bereits fertig, {len(experiments)} werden eingestellt.")

    if dry_run:
        for e in experiments:
            print("  ", e["task_id"])
        return

    if not skip_code:
        package = build_code_package()
        s3.put_object(Bucket=bucket, Key=aws.CODE_KEY, Body=package)
        print(f"Code-Paket hochgeladen: s3://{bucket}/{aws.CODE_KEY} ({len(package) / 1e6:.1f} MB)")

    sqs = sess.client("sqs")
    send_experiments(sqs, aws.queue_url(sqs), experiments)
    print("Fertig. Worker starten mit: python -m cloud.launch_workers --count N")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-code", action="store_true", help="Code-Paket nicht neu hochladen")
    args = parser.parse_args()
    main(args.max_experiments, args.dry_run, args.skip_code)
