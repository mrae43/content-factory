"""ShortComposerAgent — composes per-scene assets into a final MP4.

5-step pipeline: pre-flight → concurrent download → script compilation
(ASS subtitles) → atomic FFmpeg → clean-up & ship.
"""

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple, Type

import aiohttp
from pydantic import BaseModel

from app.services.short_config import (
    KB_MOTION_PRESETS,
    PLATFORM_ASPECT_RATIOS_SHORT,
)
from app.services.subtitles import generate_ass_file
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


def _match_words_to_scenes(
    vocal_alignment_data: List[Dict], scenes: List[Dict]
) -> List[Tuple[int, List[Dict]]]:
    """Match word-level alignment entries to scenes by word count."""
    word_index = 0
    groups: List[Tuple[int, List[Dict]]] = []
    for scene_idx, scene in enumerate(scenes):
        narration = scene.get("narration_text", "").strip()
        clean = narration.rstrip(".!?,; ")
        expected = len(clean.split()) if clean else 0
        matched: List[Dict] = []
        for _ in range(expected):
            if word_index < len(vocal_alignment_data):
                matched.append(vocal_alignment_data[word_index])
                word_index += 1
        groups.append((scene_idx, matched))
    if word_index < len(vocal_alignment_data) and groups:
        last = len(groups) - 1
        groups[last] = (
            groups[last][0],
            groups[last][1] + vocal_alignment_data[word_index:],
        )
    return groups


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
            async with aiohttp.ClientSession() as session:
                alignment_data = await self._download_json(session, vocal_alignment_url)
                if not alignment_data:
                    return AgentResult(
                        status=AgentActionStatus.ERROR,
                        payload={},
                        reasoning="Empty or missing vocal_alignment_data",
                        confidence_score=0.0,
                    )

                voiceover_path = assets_dir / "voiceover.mp3"
                try:
                    await self._download_file(session, voiceover_url, voiceover_path)
                except Exception as exc:
                    return AgentResult(
                        status=AgentActionStatus.ERROR,
                        payload={},
                        reasoning=f"Voiceover download failed: {exc}",
                        confidence_score=0.0,
                    )

                local_paths: List[Dict[str, Any]] = []
                for scene in scenes:
                    sn = scene.get("scene_number", 0)
                    atype = scene.get("asset_type")
                    if atype == "video_clip":
                        url = scene["video_url"]
                        dest = assets_dir / f"scene_{sn:02d}.mp4"
                    else:
                        url = scene["image_url"]
                        dest = assets_dir / f"scene_{sn:02d}.png"
                    try:
                        await self._download_file(session, url, dest)
                    except Exception as exc:
                        return AgentResult(
                            status=AgentActionStatus.ERROR,
                            payload={},
                            reasoning=(f"Scene {sn} asset download failed: {exc}"),
                            confidence_score=0.0,
                        )
                    local_paths.append(
                        {
                            "scene_number": sn,
                            "asset_type": atype,
                            "local_path": str(dest),
                        }
                    )

            # ── Step 3: Script compilation (ASS subtitles) ───────────────
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

            # ── Step 4: Atomic FFmpeg ────────────────────────────────────
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
            )

            if not success:
                return AgentResult(
                    status=AgentActionStatus.ERROR,
                    payload={},
                    reasoning="FFmpeg composition failed",
                    confidence_score=0.0,
                )

            # ── Step 5: Clean up & ship ────────────────────────────────────
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

    async def _download_json(
        self, session: aiohttp.ClientSession, url: str
    ) -> Optional[List[Dict]]:
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.warning("Download JSON HTTP %d: %s", resp.status, url)
                    return None
                return await resp.json()
        except Exception as exc:
            logger.warning("Download JSON failed: %s", exc)
            return None

    async def _download_file(
        self,
        session: aiohttp.ClientSession,
        url: str,
        dest: Path,
    ) -> None:
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                content = await resp.read()
                dest.write_bytes(content)
        except Exception as exc:
            logger.warning("Download file failed: %s", exc)
            raise

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
    ) -> bool:
        cmd = ["ffmpeg", "-y"]

        # Scene inputs
        for lp in local_paths:
            cmd.extend(["-i", lp["local_path"]])

        # Voiceover input
        cmd.extend(["-i", voiceover_path])

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

        # Audio
        audio_idx = len(scenes)
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
