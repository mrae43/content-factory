from app.core.config import settings
from app.services.tools import Tool


def get_storage():
    if settings.storage_backend == "local":
        from app.storage.local import LocalStorage

        return LocalStorage()

    if settings.storage_backend == "s3":
        from app.storage.s3 import S3Storage

        return S3Storage()

    raise ValueError(f"Unknown storage backend: {settings.storage_backend}")


def make_upload_image_tool() -> Tool:
    async def _upload(
        file_bytes: bytes,
        filename: str,
        folder: str = "",
        content_type: str = "image/png",
    ) -> str:
        storage = get_storage()
        return storage.upload_image(file_bytes, filename, folder, content_type)

    return Tool(
        name="upload_image",
        description="Upload an image to the configured storage backend.",
        callable=_upload,
        permissions={"CarouselImageAgent", "*"},
    )
