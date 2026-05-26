from app.core.config import settings


def get_storage():
    if settings.storage_backend == "local":
        from app.storage.local import LocalStorage

        return LocalStorage()

    if settings.storage_backend == "s3":
        from app.storage.s3 import S3Storage

        return S3Storage()

    raise ValueError(f"Unknown storage backend: {settings.storage_backend}")
