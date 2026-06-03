from pathlib import Path

from app.core.config import settings

UPLOAD_DIR = Path(settings.image_storage_path)
VIDEO_UPLOAD_DIR = Path("static/videos")
VOICEOVER_UPLOAD_DIR = Path("static/voiceovers")
MUSIC_UPLOAD_DIR = Path("static/music")


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

    def upload_voiceover(
        self,
        file_bytes: bytes,
        filename: str,
        folder: str = "",
        content_type: str = "audio/mpeg",
    ) -> str:
        target = VOICEOVER_UPLOAD_DIR / folder if folder else VOICEOVER_UPLOAD_DIR
        target.mkdir(parents=True, exist_ok=True)
        path = target / filename
        path.write_bytes(file_bytes)
        prefix = f"{folder}/" if folder else ""
        return f"/api/proxy/voiceovers/{prefix}{filename}"

    def download_file(self, url_or_path: str) -> bytes:
        if url_or_path.startswith("/api/proxy/images/"):
            rel = url_or_path[len("/api/proxy/images/") :]
            path = UPLOAD_DIR / rel
        elif url_or_path.startswith("/api/proxy/videos/"):
            rel = url_or_path[len("/api/proxy/videos/") :]
            path = VIDEO_UPLOAD_DIR / rel
        elif url_or_path.startswith("/api/proxy/voiceovers/"):
            rel = url_or_path[len("/api/proxy/voiceovers/") :]
            path = VOICEOVER_UPLOAD_DIR / rel
        elif url_or_path.startswith("/api/proxy/music/"):
            rel = url_or_path[len("/api/proxy/music/") :]
            path = MUSIC_UPLOAD_DIR / rel
        else:
            path = Path(url_or_path)
        return path.read_bytes()
