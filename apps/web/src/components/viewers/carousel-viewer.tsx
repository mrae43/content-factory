"use client";

import type { CarouselFormatPayload, PlatformEnum } from "@content-factory/shared-types";

interface CarouselViewerProps {
  payload: CarouselFormatPayload;
  platform?: PlatformEnum | null;
}

const charLimits: Record<string, number> = {
  twitter: 280,
  linkedin: 700,
  instagram: 2200,
  youtube: 5000,
};

export function CarouselViewer({ payload, platform }: CarouselViewerProps) {
  const limit = (platform && charLimits[platform]) ?? 500;
  const isCta = (slideNum: number) => slideNum === payload.slides.length;

  return (
    <div className="space-y-4">
      <div className="space-y-1">
        <h3 className="font-heading text-lg font-semibold">{payload.thread_title}</h3>
        {payload.hashtags.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {payload.hashtags.map((h) => (
              <span
                key={h}
                className="inline-flex items-center rounded-[4px] bg-primary/10 px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-[0.05em] text-primary"
              >
                #{h}
              </span>
            ))}
          </div>
        )}
      </div>

      {payload.slides.map((slide) => (
        <div
          key={slide.slide_number}
          className={`relative rounded-lg border p-5 space-y-3 ${
            isCta(slide.slide_number)
              ? "border-primary/30 bg-primary/5"
              : "border-border bg-card"
          }`}
        >
          <div className="flex items-start justify-between gap-3">
            <span className="font-heading text-2xl font-bold text-primary leading-none">
              {String(slide.slide_number).padStart(2, "0")}
            </span>
            {slide.hook_type && (
              <span className="inline-flex items-center rounded-[4px] bg-primary/10 px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-[0.05em] text-primary">
                {slide.hook_type}
              </span>
            )}
          </div>

          <p className="text-[15px] leading-[1.6] whitespace-pre-wrap">
            {slide.text}
          </p>

          {slide.visual_description && (
            <p className="text-xs italic text-muted-foreground">
              {slide.visual_description}
            </p>
          )}

          <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
            {slide.sources_used && slide.sources_used.length > 0 ? (
              <p className="text-xs text-muted-foreground">
                Sources: {slide.sources_used.join(", ")}
              </p>
            ) : (
              <span />
            )}
            <span
              className={`text-xs tabular-nums ${
                slide.text.length >= limit
                  ? "text-destructive font-semibold"
                  : slide.text.length >= limit * 0.9
                    ? "text-warning"
                    : "text-muted-foreground"
              }`}
            >
              {slide.text.length} / {limit}
            </span>
          </div>
        </div>
      ))}

      {payload.cta_slide && (
        <div className="rounded-lg border border-primary/30 bg-primary/5 p-5">
          <p className="text-xs font-semibold uppercase tracking-wide text-primary mb-1.5">
            Call to Action
          </p>
          <p className="text-sm">{payload.cta_slide}</p>
        </div>
      )}

      {payload.char_limit_violations && payload.char_limit_violations.length > 0 && (
        <div className="rounded-md border border-warning/30 bg-warning/5 p-3 text-xs text-warning space-y-1">
          <p className="font-semibold uppercase tracking-wide">Character Limit Warnings</p>
          <ul className="list-disc pl-4 space-y-0.5">
            {payload.char_limit_violations.map((v, i) => (
              <li key={i}>{v}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
