from typing import Any, Dict

from app.core.config import settings
from app.services.tools import Tool

_storage_cache: Dict[str, Any] = {}


def get_storage():
    backend = settings.storage_backend
    if backend not in _storage_cache:
        if backend == "local":
            from app.storage.local import LocalStorage

            _storage_cache[backend] = LocalStorage()
        elif backend == "s3":
            from app.storage.s3 import S3Storage

            _storage_cache[backend] = S3Storage()
        else:
            raise ValueError(f"Unknown storage backend: {settings.storage_backend}")
    return _storage_cache[backend]


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


def make_upload_video_tool() -> Tool:
    async def _upload(
        file_bytes: bytes,
        filename: str,
        folder: str = "",
        content_type: str = "video/mp4",
    ) -> str:
        storage = get_storage()
        return storage.upload_video(file_bytes, filename, folder, content_type)

    return Tool(
        name="upload_video",
        description="Upload a video to the configured storage backend.",
        callable=_upload,
        permissions={"VideoGeneratorAgent", "*"},
    )
