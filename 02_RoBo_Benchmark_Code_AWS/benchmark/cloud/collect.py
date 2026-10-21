# cloud/collect.py
#
# Zeigt den Fortschritt (Queue, DLQ, fertige Experimente) und fuehrt die Ergebnis-CSVs
# aller Experimente aus S3 zu einer CSV zusammen - im selben Format wie main.py.
# Danach wird wie gewohnt mit export_excel die Excel-Datei erstellt.
#
# Aufruf aus dem Ordner benchmark/:
#     python -m cloud.collect --status       # nur Fortschritt anzeigen
#     python -m cloud.collect                # herunterladen -> results/raw/<RUN_NAME>.csv + .xlsx

import argparse
import io
from pathlib import Path

import pandas as pd

from preprocessing import config
from postprocessing.excel_export import export_excel
from cloud import aws_settings as aws


def list_keys(s3, bucket, prefix):
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return keys


def print_status(sess, bucket):
    sqs = sess.client("sqs")
    names = ["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"]
    q = sqs.get_queue_attributes(QueueUrl=aws.queue_url(sqs), AttributeNames=names)["Attributes"]
    dlq = sqs.get_queue_attributes(QueueUrl=aws.queue_url(sqs, aws.DLQ_NAME), AttributeNames=names)["Attributes"]

    s3 = sess.client("s3")
    n_done = len(list_keys(s3, bucket, f"{aws.RUN_NAME}/tasks/"))
    n_errors = len(list_keys(s3, bucket, f"{aws.RUN_NAME}/errors/"))

    print(f"Lauf '{aws.RUN_NAME}'")
    print(f"  wartend:              {q['ApproximateNumberOfMessages']}")
    print(f"  in Bearbeitung:       {q['ApproximateNumberOfMessagesNotVisible']}")
    print(f"  fertig (S3):          {n_done}")
    print(f"  Fehlerprotokolle:     {n_errors}  (s3://{bucket}/{aws.RUN_NAME}/errors/)")
    print(f"  endgueltig gescheitert (DLQ): {dlq['ApproximateNumberOfMessages']}")


def collect(sess, bucket, output_path):
    s3 = sess.client("s3")
    keys = list_keys(s3, bucket, f"{aws.RUN_NAME}/tasks/")
    print(f"Lade {len(keys)} Ergebnisdateien ...")

    frames = []
    for i, key in enumerate(keys, start=1):
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        frames.append(pd.read_csv(io.BytesIO(body)))
        if i % 100 == 0:
            print(f"  {i}/{len(keys)}")
    if not frames:
        print("Noch keine Ergebnisse vorhanden.")
        return None

    df = pd.concat(frames, ignore_index=True).sort_values(["task_id", "iteration"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"CSV: {output_path} ({len(df)} Zeilen, {df['task_id'].nunique()} Experimente)")
    print(f"Excel: {export_excel(output_path)}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    bucket = aws.require_bucket()
    sess = aws.session()
    print_status(sess, bucket)
    if not args.status:
        output_path = args.output or config.RESULTS_DIR / "raw" / f"{aws.RUN_NAME}.csv"
        collect(sess, bucket, Path(output_path))


if __name__ == "__main__":
    main()
