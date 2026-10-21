# cloud/launch_workers.py
#
# Startet EC2-Instanzen, die sich selbst einrichten (ec2_user_data.sh), die Queue
# abarbeiten und sich danach selbst beenden (terminieren).
#
# Aufruf aus dem Ordner benchmark/:
#     python -m cloud.launch_workers --count 4 --instance-type c7i.4xlarge --spot
#     python -m cloud.launch_workers --list         # laufende Worker anzeigen
#     python -m cloud.launch_workers --terminate    # alle Worker sofort beenden

import argparse
import base64
from pathlib import Path

from cloud import aws_settings as aws


TAG = {"Key": "Project", "Value": "robo-benchmark-worker"}
AMI_PARAMETER = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"


def build_user_data(idle_minutes):
    script = (Path(__file__).parent / "ec2_user_data.sh").read_text(encoding="utf-8")
    replacements = {
        "__REGION__": aws.REGION,
        "__BUCKET__": aws.require_bucket(),
        "__QUEUE_NAME__": aws.QUEUE_NAME,
        "__RUN_NAME__": aws.RUN_NAME,
        "__CODE_KEY__": aws.CODE_KEY,
        "__IDLE_MINUTES__": str(idle_minutes),
    }
    for placeholder, value in replacements.items():
        script = script.replace(placeholder, value)
    return script.replace("\r\n", "\n")


def launch(ec2, ssm, count, instance_type, spot, idle_minutes, disk_gb, key_name):
    ami = ssm.get_parameter(Name=AMI_PARAMETER)["Parameter"]["Value"]
    params = dict(
        ImageId=ami,
        InstanceType=instance_type,
        MinCount=count,
        MaxCount=count,
        IamInstanceProfile={"Name": aws.ROLE_NAME},
        InstanceInitiatedShutdownBehavior="terminate",
        UserData=base64.b64encode(build_user_data(idle_minutes).encode()).decode(),
        BlockDeviceMappings=[{"DeviceName": "/dev/xvda",
                              "Ebs": {"VolumeSize": disk_gb, "VolumeType": "gp3",
                                      "DeleteOnTermination": True}}],
        TagSpecifications=[{"ResourceType": "instance",
                            "Tags": [TAG, {"Key": "Name", "Value": f"robo-worker-{aws.RUN_NAME}"}]}],
    )
    if key_name:
        params["KeyName"] = key_name
    if spot:
        # Spot ist meist 60-70 % guenstiger; bei Unterbrechung kommt die Nachricht zurueck in die Queue
        params["InstanceMarketOptions"] = {"MarketType": "spot",
                                           "SpotOptions": {"SpotInstanceType": "one-time",
                                                           "InstanceInterruptionBehavior": "terminate"}}

    instances = ec2.run_instances(**params)["Instances"]
    for inst in instances:
        print(f"Gestartet: {inst['InstanceId']} ({instance_type}{', spot' if spot else ''})")


def worker_instances(ec2):
    pages = ec2.get_paginator("describe_instances").paginate(Filters=[
        {"Name": f"tag:{TAG['Key']}", "Values": [TAG["Value"]]},
        {"Name": "instance-state-name", "Values": ["pending", "running", "stopping"]},
    ])
    return [inst for page in pages for res in page["Reservations"] for inst in res["Instances"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--instance-type", default="c7i.2xlarge")
    parser.add_argument("--spot", action="store_true")
    parser.add_argument("--idle-minutes", type=float, default=10)
    parser.add_argument("--disk-gb", type=int, default=30)
    parser.add_argument("--key-name", default=None, help="optional: EC2-Schluesselpaar fuer SSH")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--terminate", action="store_true")
    args = parser.parse_args()

    sess = aws.session()
    ec2 = sess.client("ec2")

    if args.list or args.terminate:
        instances = worker_instances(ec2)
        for inst in instances:
            print(f"{inst['InstanceId']}  {inst['InstanceType']}  {inst['State']['Name']}  {inst['LaunchTime']}")
        if args.terminate and instances:
            ec2.terminate_instances(InstanceIds=[i["InstanceId"] for i in instances])
            print(f"{len(instances)} Instanzen werden beendet.")
        return

    launch(ec2, sess.client("ssm"), args.count, args.instance_type, args.spot,
           args.idle_minutes, args.disk_gb, args.key_name)


if __name__ == "__main__":
    main()
