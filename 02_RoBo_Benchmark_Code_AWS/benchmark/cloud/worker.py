# cloud/worker.py
#
# Holt Experimente aus der SQS-Queue, rechnet sie mit dem unveraenderten
# run_single_experiment aus main_processing/experiment_runner.py und legt die Ergebnisse
# als eigene CSV pro Experiment in S3 ab (s3://BUCKET/RUN_NAME/tasks/). Die Spalten sind
# dieselben wie in der lokalen results.csv.
# Laeuft auf den EC2-Instanzen (Start ueber cloud/ec2_user_data.sh), funktioniert aber
# genauso lokal.
#
# Pro CPU-Kern laeuft ein Prozess mit genau einem Torch-Thread, damit sich die
# parallelen Experimente nicht gegenseitig die Kerne wegnehmen.
#
# Ablauf pro Nachricht:
#   1. Nachricht abholen (fuer andere Worker unsichtbar, Heartbeat verlaengert das)
#   2. Experiment rechnen
#   3. ok     -> CSV nach S3, Nachricht loeschen
#      Fehler -> Traceback nach S3 (errors/), Nachricht NICHT loeschen; sie kommt nach
#                kurzer Zeit wieder, nach MAX_RECEIVE_COUNT Versuchen -> DLQ
#
# Aufruf aus dem Ordner benchmark/:
#     python -m cloud.worker                     # ein Prozess pro CPU-Kern
#     python -m cloud.worker --processes 2 --idle-minutes 5

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import threading
import time
import traceback

from cloud import aws_settings as aws


def _heartbeat(sqs, url, receipt, stop):
    # Haelt die Nachricht unsichtbar, solange das Experiment rechnet
    while not stop.wait(aws.HEARTBEAT_INTERVAL_S):
        try:
            sqs.change_message_visibility(QueueUrl=url, ReceiptHandle=receipt,
                                          VisibilityTimeout=aws.VISIBILITY_TIMEOUT_S)
        except Exception as e:
            print(f"Heartbeat fehlgeschlagen: {e}", flush=True)


def _rows_to_csv(experiment, rows):
    # Gleiche Spaltenreihenfolge wie run_experiments beim lokalen Lauf
    import pandas as pd
    from main_processing.experiment_runner import RESULT_COLUMNS

    columns = list(experiment.keys()) + RESULT_COLUMNS
    return pd.DataFrame(rows).reindex(columns=columns).to_csv(index=False)


def worker_loop(worker_index, idle_minutes):
    # Erst im Kindprozess importieren und Threads begrenzen
    import torch
    torch.set_num_threads(1)

    from main_processing.experiment_runner import run_single_experiment

    bucket = aws.require_bucket()
    sess = aws.session()
    sqs, s3 = sess.client("sqs"), sess.client("s3")
    url = aws.queue_url(sqs)
    name = f"[w{worker_index}]"

    idle_since = time.time()
    while True:
        messages = sqs.receive_message(QueueUrl=url, MaxNumberOfMessages=1, WaitTimeSeconds=20,
                                       AttributeNames=["ApproximateReceiveCount"]).get("Messages", [])
        if not messages:
            if time.time() - idle_since > idle_minutes * 60:
                print(f"{name} Queue seit {idle_minutes} min leer - beende.", flush=True)
                return
            continue

        message = messages[0]
        receipt = message["ReceiptHandle"]
        experiment = json.loads(message["Body"])
        task_id = experiment["task_id"]
        attempt = message["Attributes"].get("ApproximateReceiveCount", "1")
        print(f"{name} Start {task_id} (Versuch {attempt})", flush=True)

        stop = threading.Event()
        heartbeat = threading.Thread(target=_heartbeat, args=(sqs, url, receipt, stop), daemon=True)
        heartbeat.start()
        start = time.time()
        try:
            rows = run_single_experiment(experiment)
            s3.put_object(Bucket=bucket, Key=aws.task_key(task_id),
                          Body=_rows_to_csv(experiment, rows).encode("utf-8"))
            sqs.delete_message(QueueUrl=url, ReceiptHandle=receipt)
            print(f"{name} ok {task_id} ({time.time() - start:.1f}s)", flush=True)
        except KeyboardInterrupt:
            # Sofort wieder freigeben, damit ein anderer Worker uebernimmt
            sqs.change_message_visibility(QueueUrl=url, ReceiptHandle=receipt, VisibilityTimeout=0)
            raise
        except Exception as e:
            traceback.print_exc()
            s3.put_object(Bucket=bucket, Key=aws.error_key(task_id),
                          Body=f"Versuch {attempt}\n{traceback.format_exc()}".encode("utf-8"))
            # Nicht sofort neu versuchen, aber auch nicht die vollen 15 min warten
            sqs.change_message_visibility(QueueUrl=url, ReceiptHandle=receipt, VisibilityTimeout=60)
            print(f"{name} FEHLER {task_id}: {type(e).__name__}: {e}", flush=True)
        finally:
            stop.set()
        idle_since = time.time()


def main(processes=None, idle_minutes=10, shutdown=False):
    processes = processes or os.cpu_count() or 1
    print(f"Starte {processes} Worker-Prozesse (Lauf '{aws.RUN_NAME}').", flush=True)

    ctx = mp.get_context("spawn")
    workers = [ctx.Process(target=worker_loop, args=(i, idle_minutes)) for i in range(processes)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    print("Alle Worker beendet.", flush=True)
    if shutdown:
        # Die Instanz wird mit InstanceInitiatedShutdownBehavior=terminate gestartet
        # -> Herunterfahren beendet sie, es fallen keine Kosten mehr an
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=None, help="Standard: Anzahl CPU-Kerne")
    parser.add_argument("--idle-minutes", type=float, default=10,
                        help="Beenden, wenn die Queue so lange leer ist")
    parser.add_argument("--shutdown", action="store_true",
                        help="Instanz am Ende herunterfahren (nur auf EC2 sinnvoll)")
    args = parser.parse_args()
    main(args.processes, args.idle_minutes, args.shutdown)
