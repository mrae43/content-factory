from pathlib import Path

from app.core.config import settings

UPLOAD_DIR = Path(settings.image_storage_path)


class LocalStorage:
    def upload_image(self, file_bytes: bytes, filename: str) -> str:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        path = UPLOAD_DIR / filename
        path.write_bytes(file_bytes)
        return f"/static/carousel_images/{filename}"

    def delete_image(self, filename: str) -> None:
        path = UPLOAD_DIR / filename
        if path.exists():
            path.unlink()
