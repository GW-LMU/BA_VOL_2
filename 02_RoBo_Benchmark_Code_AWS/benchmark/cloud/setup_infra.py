# cloud/setup_infra.py
#
# Legt einmalig die AWS-Infrastruktur an (mehrfach ausfuehren ist unschaedlich):
#   - S3-Bucket fuer Code-Paket und Ergebnisse
#   - SQS-Queue fuer die Experimente + Dead-Letter-Queue
#   - IAM-Rolle + Instanzprofil, damit die EC2-Worker Queue und Bucket nutzen duerfen
#
# Aufruf aus dem Ordner benchmark/ (AWS-Zugangsdaten muessen konfiguriert sein):
#     python -m cloud.setup_infra

import json

from botocore.exceptions import ClientError

from cloud import aws_settings as aws


def create_bucket(s3, bucket):
    try:
        if aws.REGION == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(Bucket=bucket,
                             CreateBucketConfiguration={"LocationConstraint": aws.REGION})
        print(f"Bucket angelegt: {bucket}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
            raise
        print(f"Bucket existiert bereits: {bucket}")


def create_queues(sqs):
    # create_queue ist idempotent, solange die Attribute gleich bleiben
    dlq_url = sqs.create_queue(QueueName=aws.DLQ_NAME,
                               Attributes={"MessageRetentionPeriod": "1209600"})["QueueUrl"]
    dlq_arn = sqs.get_queue_attributes(QueueUrl=dlq_url, AttributeNames=["QueueArn"])["Attributes"]["QueueArn"]

    url = sqs.create_queue(QueueName=aws.QUEUE_NAME, Attributes={
        "VisibilityTimeout": str(aws.VISIBILITY_TIMEOUT_S),
        "MessageRetentionPeriod": "1209600",            # 14 Tage (Maximum)
        "ReceiveMessageWaitTimeSeconds": "20",          # Long Polling
        "RedrivePolicy": json.dumps({"deadLetterTargetArn": dlq_arn,
                                     "maxReceiveCount": str(aws.MAX_RECEIVE_COUNT)}),
    })["QueueUrl"]
    arn = sqs.get_queue_attributes(QueueUrl=url, AttributeNames=["QueueArn"])["Attributes"]["QueueArn"]
    print(f"Queue: {url}\nDLQ:   {dlq_url}")
    return arn, dlq_arn


def create_worker_role(iam, bucket, queue_arn, dlq_arn):
    trust = {
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"},
                       "Action": "sts:AssumeRole"}],
    }
    # Nur die noetigsten Rechte: diese Queue und dieser Bucket
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow",
             "Action": ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:ChangeMessageVisibility",
                        "sqs:GetQueueUrl", "sqs:GetQueueAttributes"],
             "Resource": [queue_arn, dlq_arn]},
            {"Effect": "Allow",
             "Action": ["s3:GetObject", "s3:PutObject"],
             "Resource": f"arn:aws:s3:::{bucket}/*"},
            {"Effect": "Allow", "Action": ["s3:ListBucket"], "Resource": f"arn:aws:s3:::{bucket}"},
        ],
    }

    try:
        iam.create_role(RoleName=aws.ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust))
        print(f"Rolle angelegt: {aws.ROLE_NAME}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "EntityAlreadyExists":
            raise
    iam.put_role_policy(RoleName=aws.ROLE_NAME, PolicyName="robo-benchmark-access",
                        PolicyDocument=json.dumps(policy))

    try:
        iam.create_instance_profile(InstanceProfileName=aws.ROLE_NAME)
        iam.add_role_to_instance_profile(InstanceProfileName=aws.ROLE_NAME, RoleName=aws.ROLE_NAME)
        print(f"Instanzprofil angelegt: {aws.ROLE_NAME} (IAM braucht ggf. ~10 s bis es nutzbar ist)")
    except ClientError as e:
        if e.response["Error"]["Code"] != "EntityAlreadyExists":
            raise


def main():
    bucket = aws.require_bucket()
    sess = aws.session()
    create_bucket(sess.client("s3"), bucket)
    queue_arn, dlq_arn = create_queues(sess.client("sqs"))
    create_worker_role(sess.client("iam"), bucket, queue_arn, dlq_arn)
    print("Infrastruktur bereit.")


if __name__ == "__main__":
    main()
