# cloud/aws_settings.py
#
# Gemeinsame AWS-Einstellungen fuer setup_infra, submit, worker, launch_workers und collect.
# Alle Werte koennen ueber Umgebungsvariablen ueberschrieben werden (die EC2-Worker
# bekommen sie per User-Data gesetzt).

import os

import boto3


REGION = os.environ.get("ROBO_AWS_REGION", "eu-central-1")
QUEUE_NAME = os.environ.get("ROBO_QUEUE_NAME", "robo-benchmark-experiments")
DLQ_NAME = QUEUE_NAME + "-failed"           # Dead-Letter-Queue fuer wiederholt fehlgeschlagene Experimente
BUCKET = os.environ.get("ROBO_BUCKET")      # muss weltweit eindeutig sein
RUN_NAME = os.environ.get("ROBO_RUN_NAME", "results")   # Unterordner im Bucket, ein Ordner pro Benchmark-Lauf

# Rolle / Instanzprofil, mit dem die EC2-Worker auf Queue und Bucket zugreifen
ROLE_NAME = "robo-benchmark-worker"

# SQS: so lange ist eine Nachricht nach dem Abholen fuer andere Worker unsichtbar.
# Der Worker verlaengert sie laufend (Heartbeat), solange das Experiment rechnet.
# Stirbt der Worker (z. B. Spot-Unterbrechung), taucht die Nachricht danach wieder auf.
VISIBILITY_TIMEOUT_S = 900
HEARTBEAT_INTERVAL_S = 300
MAX_RECEIVE_COUNT = 3                       # danach landet die Nachricht in der DLQ

CODE_KEY = "code/benchmark.tar.gz"          # Code-Paket, das die Worker beim Start laden


def require_bucket():
    if not BUCKET:
        raise SystemExit("Umgebungsvariable ROBO_BUCKET ist nicht gesetzt (Name des S3-Buckets).")
    return BUCKET


def task_key(task_id):
    return f"{RUN_NAME}/tasks/{task_id}.csv"


def error_key(task_id):
    return f"{RUN_NAME}/errors/{task_id}.txt"


def session():
    return boto3.session.Session(region_name=REGION)


def queue_url(sqs, name=QUEUE_NAME):
    return sqs.get_queue_url(QueueName=name)["QueueUrl"]
