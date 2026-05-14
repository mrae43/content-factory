"use client";

import type { VideoFormatPayload } from "@content-factory/shared-types";

interface VideoScriptViewerProps {
  payload: VideoFormatPayload;
}

export function VideoScriptViewer({ payload }: VideoScriptViewerProps) {
  return (
    <div className="space-y-5">
      <div className="space-y-2">
        <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-muted-foreground">
          <span>
            <span className="font-semibold text-foreground">Visual Style:</span>{" "}
            {payload.visual_style}
          </span>
          {payload.audio_direction && (
            <span>
              <span className="font-semibold text-foreground">Audio Direction:</span>{" "}
              {payload.audio_direction}
            </span>
          )}
          <span>
            <span className="font-semibold text-foreground">Total Duration:</span>{" "}
            {payload.total_duration_seconds}s
          </span>
        </div>
        <div className="editorial-rule" />
      </div>

      {payload.scenes.map((scene) => (
        <div
          key={scene.scene_number}
          className="rounded-lg border border-border bg-card p-5 space-y-4"
        >
          <div className="flex items-center justify-between">
            <span className="font-heading text-base font-semibold">
              Scene {scene.scene_number}
            </span>
            <span className="text-xs tabular-nums text-muted-foreground">
              {scene.duration_seconds}s
            </span>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="rounded-md bg-muted p-3 space-y-1">
              <p className="text-[11px] font-semibold uppercase tracking-[0.05em] text-primary">
                Visual
              </p>
              <p className="text-sm leading-relaxed">{scene.visual_prompt}</p>
            </div>
            <div className="rounded-md bg-muted p-3 space-y-1">
              <p className="text-[11px] font-semibold uppercase tracking-[0.05em] text-primary">
                Narration
              </p>
              <p className="text-sm leading-relaxed italic">
                &ldquo;{scene.narration_text}&rdquo;
              </p>
            </div>
          </div>

          {scene.audio_cue && (
            <p className="text-xs italic text-muted-foreground">
              Audio: {scene.audio_cue}
            </p>
          )}
        </div>
      ))}
    </div>
  );
}
