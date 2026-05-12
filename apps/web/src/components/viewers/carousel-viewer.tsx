"use client";

import type { CarouselFormatPayload } from "@content-factory/shared-types";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface CarouselViewerProps {
  payload: CarouselFormatPayload;
}

export function CarouselViewer({ payload }: CarouselViewerProps) {
  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold">{payload.thread_title}</h3>
        {payload.hashtags.length > 0 && (
          <p className="text-sm text-muted-foreground">
            {payload.hashtags.map((h) => `#${h}`).join(" ")}
          </p>
        )}
      </div>

      {payload.slides.map((slide) => (
        <Card key={slide.slide_number}>
          <CardContent className="pt-4 space-y-2">
            <div className="flex items-center gap-2">
              <Badge variant="outline">{slide.slide_number}</Badge>
              {slide.hook_type && (
                <span className="text-xs text-muted-foreground">
                  {slide.hook_type}
                </span>
              )}
            </div>
            <p className="whitespace-pre-wrap text-sm">{slide.text}</p>
            {slide.visual_prompt && (
              <details className="text-xs text-muted-foreground">
                <summary className="cursor-pointer">Visual Prompt</summary>
                <p className="mt-1">{slide.visual_prompt}</p>
              </details>
            )}
            {slide.sources_used && slide.sources_used.length > 0 && (
              <p className="text-xs text-muted-foreground">
                Sources: {slide.sources_used.join(", ")}
              </p>
            )}
          </CardContent>
        </Card>
      ))}

      {payload.cta_slide && (
        <Card className="border-primary/30">
          <CardContent className="pt-4">
            <p className="text-sm font-medium">CTA Slide</p>
            <p className="text-sm">{payload.cta_slide}</p>
          </CardContent>
        </Card>
      )}

      {payload.char_limit_violations && payload.char_limit_violations.length > 0 && (
        <div className="rounded-md bg-yellow-50 p-3 text-xs text-yellow-800">
          <p className="font-medium">Character Limit Warnings</p>
          <ul className="mt-1 list-disc pl-4">
            {payload.char_limit_violations.map((v, i) => (
              <li key={i}>{v}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
