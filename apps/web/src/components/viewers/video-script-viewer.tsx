"use client";

import type { VideoFormatPayload } from "@content-factory/shared-types";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

interface VideoScriptViewerProps {
  payload: VideoFormatPayload;
}

export function VideoScriptViewer({ payload }: VideoScriptViewerProps) {
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-4 text-sm">
        <div>
          <span className="font-medium">Visual Style:</span>{" "}
          {payload.visual_style}
        </div>
        {payload.audio_direction && (
          <div>
            <span className="font-medium">Audio Direction:</span>{" "}
            {payload.audio_direction}
          </div>
        )}
        <div>
          <span className="font-medium">Total Duration:</span>{" "}
          {payload.total_duration_seconds}s
        </div>
      </div>

      <Separator />

      {payload.scenes.map((scene) => (
        <Card key={scene.scene_number}>
          <CardContent className="pt-4 space-y-2">
            <div className="flex items-center gap-2">
              <Badge variant="outline">Scene {scene.scene_number}</Badge>
              <span className="text-xs text-muted-foreground">
                {scene.duration_seconds}s
              </span>
            </div>
            <div className="space-y-1 text-sm">
              <p>
                <span className="font-medium">Narration:</span>{" "}
                {scene.narration_text}
              </p>
              <p>
                <span className="font-medium">Visual:</span>{" "}
                {scene.visual_prompt}
              </p>
              {scene.audio_cue && (
                <p>
                  <span className="font-medium">Audio:</span>{" "}
                  {scene.audio_cue}
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
