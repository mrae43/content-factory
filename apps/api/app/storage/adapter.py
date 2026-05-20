from app.core.config import settings


def get_storage():
    if settings.storage_backend == "local":
        from app.storage.local import LocalStorage

        return LocalStorage()

    raise ValueError(f"Unknown storage backend: {settings.storage_backend}")
