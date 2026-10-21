#!/bin/bash
# cloud/ec2_user_data.sh
#
# Startskript der EC2-Worker (Amazon Linux 2023). launch_workers.py ersetzt die
# Platzhalter __...__ und uebergibt das Skript als User-Data.
# Log auf der Instanz: /var/log/robo-worker.log
set -euxo pipefail
exec > /var/log/robo-worker.log 2>&1

export ROBO_AWS_REGION="__REGION__"
export ROBO_BUCKET="__BUCKET__"
export ROBO_QUEUE_NAME="__QUEUE_NAME__"
export ROBO_RUN_NAME="__RUN_NAME__"

# Bei Fehlern in der Einrichtung trotzdem herunterfahren (= terminieren), damit keine Kosten anfallen
trap 'shutdown -h now' ERR

dnf install -y python3.11 python3.11-pip

mkdir -p /opt/robo && cd /opt/robo
aws s3 cp "s3://${ROBO_BUCKET}/__CODE_KEY__" benchmark.tar.gz --region "${ROBO_AWS_REGION}"
tar xzf benchmark.tar.gz

python3.11 -m venv venv
./venv/bin/pip install --upgrade pip
# CPU-Version von Torch (ohne CUDA, deutlich kleiner)
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
./venv/bin/pip install -r benchmark/requirements.txt -r benchmark/cloud/requirements-aws.txt

cd benchmark
../venv/bin/python -m cloud.worker --idle-minutes __IDLE_MINUTES__ --shutdown
