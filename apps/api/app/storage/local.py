from pathlib import Path

from app.core.config import settings

UPLOAD_DIR = Path(settings.image_storage_path)
VIDEO_UPLOAD_DIR = Path("static/videos")


class LocalStorage:
    def upload_image(
        self,
        file_bytes: bytes,
        filename: str,
        folder: str = "",
        content_type: str = "image/png",
    ) -> str:
        target = UPLOAD_DIR / folder if folder else UPLOAD_DIR
        target.mkdir(parents=True, exist_ok=True)
        path = target / filename
        path.write_bytes(file_bytes)
        prefix = f"{folder}/" if folder else ""
        return f"/api/proxy/images/{prefix}{filename}"

    def delete_image(self, filename: str, folder: str = "") -> None:
        target = UPLOAD_DIR / folder if folder else UPLOAD_DIR
        path = target / filename
        if path.exists():
            path.unlink()

    def upload_video(
        self,
        file_bytes: bytes,
        filename: str,
        folder: str = "",
        content_type: str = "video/mp4",
    ) -> str:
        target = VIDEO_UPLOAD_DIR / folder if folder else VIDEO_UPLOAD_DIR
        target.mkdir(parents=True, exist_ok=True)
        path = target / filename
        path.write_bytes(file_bytes)
        prefix = f"{folder}/" if folder else ""
        return f"/api/proxy/videos/{prefix}{filename}"
