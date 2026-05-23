import boto3
from botocore.config import Config as BotoConfig
from app.core.config import settings


class S3Storage:
    def __init__(self):
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
            config=BotoConfig(signature_version="s3v4"),
        )
        self.bucket = settings.s3_bucket_images
        self.public_url = settings.s3_public_url.rstrip("/")

    def upload_image(self, file_bytes: bytes, filename: str, folder: str = "") -> str:
        key = f"{folder}/{filename}" if folder else filename

        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:
            self.client.create_bucket(Bucket=self.bucket)

        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=file_bytes,
            ContentType="image/png",
        )
        return f"{self.public_url}/{self.bucket}/{key}"

    def delete_image(self, filename: str, folder: str = "") -> None:
        key = f"{folder}/{filename}" if folder else filename
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
        except Exception:
            pass
