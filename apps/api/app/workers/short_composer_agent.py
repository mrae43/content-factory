"""ShortComposerAgent — composes per-scene assets into a final MP4.

5-step pipeline: pre-flight → concurrent download → script compilation
(ASS subtitles) → atomic FFmpeg → clean-up & ship.
"""

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Set, Type

import json
from pydantic import BaseModel

from app.core.config import settings
from app.services.short_config import (
    KB_MOTION_PRESETS,
    MUSIC_MOOD_MAP,
    MUSIC_VOLUME,
    PLATFORM_ASPECT_RATIOS_SHORT,
)
from app.services.subtitles import (
    _match_words_to_scenes,
    generate_ass_file,
)
from app.storage.adapter import get_storage
from app.workers.agents import AgentActionStatus, AgentResult, ServiceAgent

logger = logging.getLogger(__name__)

FPS = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare_zoompan_expr(kb_motion: str, duration: float, fps: int = FPS) -> str:
    """Substitute duration*fps placeholder with evaluated frame count."""
    total_frames = int(duration * fps)
    expr = KB_MOTION_PRESETS[kb_motion]["zoompan"]
    return expr.replace("duration*fps", str(total_frames))


def _resolve_music_url(music_mood: str) -> Optional[str]:
    """Resolve a music_mood tag to a downloadable URL, or None if unknown."""
    filename = MUSIC_MOOD_MAP.get(music_mood)
    if not filename:
        if music_mood:
            logger.warning(
                "Unknown music_mood '%s' — skipping background music", music_mood
            )
        return None
    if settings.storage_backend == "s3":
        return (
            f"{settings.s3_public_url}/{settings.s3_bucket_music}"
            f"/music/{music_mood}/{filename}"
        )
    return f"/api/proxy/music/{music_mood}/{filename}"


def _compute_scene_durations(
    vocal_alignment_data: List[Dict], scenes: List[Dict]
) -> List[float]:
    """Derive per-scene durations from vocal alignment word timestamps."""
    groups = _match_words_to_scenes(vocal_alignment_data, scenes)
    durations: List[float] = []
    for _scene_idx, words in groups:
        if words:
            dur = words[-1].get("end", 0.0) - words[0].get("start", 0.0)
            durations.append(max(dur, 1.0))
        else:
            durations.append(3.0)
    return durations


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ShortComposerAgent(ServiceAgent):
    _required_di_tools: ClassVar[List[str]] = []
    _permissions: ClassVar[Set[str]] = {"ShortComposerAgent"}
    input_schema: ClassVar[Optional[Type[BaseModel]]] = None

    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        format_payload = context.get("format_payload", {})
        job_id = context.get("job_id")
        platform = context.get("platform", "tiktok")
        voiceover_url = context.get("voiceover_url")
        vocal_alignment_url = context.get("vocal_alignment_url")

        if not isinstance(format_payload, dict):
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="format_payload must be a dict",
                confidence_score=0.0,
            )

        scenes = format_payload.get("scenes", [])
        if not scenes:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="No scenes found in format_payload",
                confidence_score=0.0,
            )

        # ── Step 1: Pre-flight ───────────────────────────────────────────
        missing: List[str] = []
        for scene in scenes:
            sn = scene.get("scene_number", "?")
            atype = scene.get("asset_type")
            if atype == "video_clip" and not scene.get("video_url"):
                missing.append(f"scene {sn} video_url")
            elif atype == "ken_burns" and not scene.get("image_url"):
                missing.append(f"scene {sn} image_url")

        if missing:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning=f"Missing scene assets: {', '.join(missing)}",
                confidence_score=0.0,
            )

        if not voiceover_url:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="voiceover_url is required",
                confidence_score=0.0,
            )
        if not vocal_alignment_url:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="vocal_alignment_url is required",
                confidence_score=0.0,
            )

        tmp_dir = Path(f"/tmp/job_{job_id or 'standalone'}")
        assets_dir = tmp_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        try:
            # ── Step 2: Concurrent download ──────────────────────────────
            storage = get_storage()

            # Vocal alignment JSON
            try:
                alignment_bytes = await asyncio.to_thread(
                    storage.download_file, vocal_alignment_url
                )
                alignment_data = json.loads(alignment_bytes)
            except Exception as exc:
                return AgentResult(
                    status=AgentActionStatus.ERROR,
                    payload={},
                    reasoning=f"Vocal alignment download failed: {exc}",
                    confidence_score=0.0,
                )

            if not alignment_data:
                return AgentResult(
                    status=AgentActionStatus.ERROR,
                    payload={},
                    reasoning="Empty or missing vocal_alignment_data",
                    confidence_score=0.0,
                )

            # Voiceover audio
            voiceover_path = assets_dir / "voiceover.mp3"
            try:
                voiceover_bytes = await asyncio.to_thread(
                    storage.download_file, voiceover_url
                )
                voiceover_path.write_bytes(voiceover_bytes)
            except Exception as exc:
                return AgentResult(
                    status=AgentActionStatus.ERROR,
                    payload={},
                    reasoning=f"Voiceover download failed: {exc}",
                    confidence_score=0.0,
                )

            # Scene assets — concurrent via asyncio.gather
            async def _download_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
                sn = scene.get("scene_number", 0)
                atype = scene.get("asset_type")
                if atype == "video_clip":
                    url = scene["video_url"]
                    dest = assets_dir / f"scene_{sn:02d}.mp4"
                else:
                    url = scene["image_url"]
                    dest = assets_dir / f"scene_{sn:02d}.png"
                data = await asyncio.to_thread(storage.download_file, url)
                dest.write_bytes(data)
                return {
                    "scene_number": sn,
                    "asset_type": atype,
                    "local_path": str(dest),
                }

            try:
                local_paths = await asyncio.gather(
                    *[_download_scene(s) for s in scenes]
                )
            except Exception as exc:
                return AgentResult(
                    status=AgentActionStatus.ERROR,
                    payload={},
                    reasoning=f"Scene asset download failed: {exc}",
                    confidence_score=0.0,
                )

            # ── Step 3: Background music (optional — graceful degradation) ──
            music_mood = format_payload.get("music_mood", "")
            music_path: Optional[Path] = None
            if music_mood:
                music_url = _resolve_music_url(music_mood)
                if music_url:
                    try:
                        music_bytes = await asyncio.to_thread(
                            storage.download_file, music_url
                        )
                        music_path = assets_dir / "background_music.mp3"
                        music_path.write_bytes(music_bytes)
                    except Exception as exc:
                        logger.warning(
                            "Background music download failed for mood '%s': %s",
                            music_mood,
                            exc,
                        )

            # ── Step 4: Script compilation (ASS subtitles) ───────────────
            subtitle_preset = format_payload.get("subtitle_preset", "CENTER_POP_YELLOW")
            try:
                ass_content = generate_ass_file(
                    alignment_data, scenes, subtitle_preset, platform
                )
            except ValueError as exc:
                return AgentResult(
                    status=AgentActionStatus.ERROR,
                    payload={},
                    reasoning=f"Subtitle generation failed: {exc}",
                    confidence_score=0.0,
                )

            ass_path = assets_dir / "subtitles.ass"
            ass_path.write_text(ass_content, encoding="utf-8")

            scene_durations = _compute_scene_durations(alignment_data, scenes)

            # ── Step 5: Atomic FFmpeg ────────────────────────────────────
            width, height = PLATFORM_ASPECT_RATIOS_SHORT.get(platform, (1080, 1920))
            output_path = tmp_dir / "final_output.mp4"
            success = await self._run_ffmpeg(
                scenes=scenes,
                scene_durations=scene_durations,
                local_paths=local_paths,
                voiceover_path=str(voiceover_path),
                ass_path=str(ass_path),
                output_path=str(output_path),
                width=width,
                height=height,
                music_path=str(music_path) if music_path else None,
            )

            if not success:
                return AgentResult(
                    status=AgentActionStatus.ERROR,
                    payload={},
                    reasoning="FFmpeg composition failed",
                    confidence_score=0.0,
                )

            # ── Step 6: Clean up & ship ────────────────────────────────────
            folder = (
                f"{context.get('device_id', '__anonymous__')}/{job_id or 'standalone'}"
            )
            storage = get_storage()
            final_url = await asyncio.to_thread(
                storage.upload_video,
                output_path.read_bytes(),
                f"final_output_{job_id or 'standalone'}.mp4",
                folder=folder,
            )

            return AgentResult(
                status=AgentActionStatus.SUCCESS,
                payload={"final_video_url": final_url},
                reasoning=f"Composed {len(scenes)} scenes into final video",
                confidence_score=1.0,
                metadata={
                    "total_scenes": len(scenes),
                    "platform": platform,
                    "resolution": f"{width}x{height}",
                },
            )

        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Internal helpers ───────────────────────────────────────────────

    async def _run_ffmpeg(
        self,
        scenes: List[Dict[str, Any]],
        scene_durations: List[float],
        local_paths: List[Dict[str, Any]],
        voiceover_path: str,
        ass_path: str,
        output_path: str,
        width: int,
        height: int,
        music_path: Optional[str] = None,
    ) -> bool:
        cmd = ["ffmpeg", "-y"]

        # Scene inputs
        for lp in local_paths:
            cmd.extend(["-i", lp["local_path"]])

        # Voiceover input
        cmd.extend(["-i", voiceover_path])

        # Background music input (optional)
        if music_path:
            cmd.extend(["-i", music_path])

        filters: List[str] = []

        for i, (scene, dur) in enumerate(zip(scenes, scene_durations)):
            if scene.get("asset_type") == "ken_burns":
                kb = scene.get("kb_motion", "zoom_in")
                zoompan_expr = _prepare_zoompan_expr(kb, dur, FPS)
                filters.append(
                    f"[{i}:v]zoompan={zoompan_expr},fps={FPS},"
                    f"scale={width}:{height},setsar=1,"
                    f"format=yuv420p[v{i}];"
                )
            else:
                filters.append(
                    f"[{i}:v]trim=duration={dur},fps={FPS},"
                    f"scale={width}:{height},setsar=1,"
                    f"format=yuv420p[v{i}];"
                )

        # Concatenate video streams
        concat_inputs = "".join(f"[v{i}]" for i in range(len(scenes)))
        filters.append(f"{concat_inputs}concat=n={len(scenes)}:v=1:a=0[concatv];")

        # Burn subtitles
        filters.append(f"[concatv]ass={ass_path}[subv];")

        # Audio — voiceover with optional background music overlay
        audio_idx = len(scenes)
        if music_path:
            music_idx = len(scenes) + 1
            filters.append(f"[{audio_idx}:a]volume=1.0[vo];")
            filters.append(f"[{music_idx}:a]volume={MUSIC_VOLUME}[bm0];")
            filters.append("[vo][bm0]amix=inputs=2:duration=first[audio];")
        else:
            filters.append(f"[{audio_idx}:a]volume=1.0[a0];")
            filters.append("[a0]acopy[audio];")

        cmd.extend(["-filter_complex", "".join(filters)])
        cmd.extend(["-map", "[subv]", "-map", "[audio]"])
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                output_path,
            ]
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="replace")[-500:]
                logger.error("FFmpeg failed (rc=%d): %s", proc.returncode, err)
                return False
            return True
        except asyncio.TimeoutError:
            logger.error("FFmpeg timed out after 300s")
            return False
        except FileNotFoundError:
            logger.error("FFmpeg binary not found")
            return False
        except Exception as exc:
            logger.error("FFmpeg execution error: %s", exc)
            return False
