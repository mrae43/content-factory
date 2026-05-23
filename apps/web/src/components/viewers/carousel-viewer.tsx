"use client";

import { useState, useEffect } from "react";
import type { CarouselFormatPayload, PlatformEnum } from "@content-factory/shared-types";
import {
  HelpCircle,
  TrendingUp,
  Quote,
  Image as Img,
  BookOpen,
  ArrowRight,
  ChevronLeft,
  ChevronRight,
  FileText,
} from "lucide-react";

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

const HOOK_TYPE_ICON_MAP: Record<string, typeof HelpCircle> = {
  question: HelpCircle,
  statistic: TrendingUp,
  quote: Quote,
  visual: Img,
  story: BookOpen,
  cta: ArrowRight,
};

const ASPECT_RATIO_MAP: Record<string, string> = {
  twitter: "2/3",
  linkedin: "4/5",
  instagram: "4/5",
  tiktok: "9/16",
  youtube: "16/9",
};

export function CarouselViewer({ payload, platform }: CarouselViewerProps) {
  const limit = (platform && charLimits[platform]) ?? 500;
  const slides = payload.slides;
  const totalSlides = slides.length;

  const [currentIndex, setCurrentIndex] = useState(0);

  const goPrev = () => setCurrentIndex((i) => Math.max(0, i - 1));
  const goNext = () => setCurrentIndex((i) => Math.min(totalSlides - 1, i + 1));

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft") {
        setCurrentIndex((i) => Math.max(0, i - 1));
      }
      if (e.key === "ArrowRight") {
        setCurrentIndex((i) => Math.min(totalSlides - 1, i + 1));
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [totalSlides]);

  const slide = slides[currentIndex];
  if (!slide) return null;

  const isCtaSlide = currentIndex === totalSlides - 1 && slide.hook_type === "cta";
  const IconComponent = HOOK_TYPE_ICON_MAP[slide.hook_type] ?? FileText;
  const aspectRatio = ASPECT_RATIO_MAP[platform ?? ""] ?? "4/5";

  return (
    <div className="space-y-4">
      <h3 className="font-heading text-lg font-semibold">{payload.thread_title}</h3>

      <div
        className="max-w-sm md:max-w-md mx-auto"
        style={{ aspectRatio }}
      >
        {slide.image_url ? (
          <div className="h-full rounded-lg border border-border overflow-hidden relative bg-black">
            <img
              src={slide.image_url}
              alt={`Slide ${slide.slide_number}`}
              className="h-full w-full object-contain"
            />
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-4 pt-12">
              <div className="flex items-center justify-between gap-2 mb-1">
                <span className="font-heading text-lg font-bold text-white leading-none">
                  {String(slide.slide_number).padStart(2, "0")}
                </span>
                <IconComponent className="h-4 w-4 text-white/80" />
              </div>
              <p className="text-sm leading-snug text-white whitespace-pre-wrap line-clamp-3">
                {slide.text}
              </p>
            </div>
          </div>
        ) : (
          <div
            className={`h-full rounded-lg border p-5 flex flex-col space-y-3 ${
              isCtaSlide
                ? "border-primary/30 bg-primary/5"
                : "border-border bg-card"
            }`}
          >
            <div className="flex items-start justify-between gap-3">
              <span className="font-heading text-2xl font-bold text-primary leading-none">
                {String(slide.slide_number).padStart(2, "0")}
              </span>
              <IconComponent
                className={`h-5 w-5 ${
                  isCtaSlide ? "text-primary" : "text-muted-foreground"
                }`}
              />
            </div>

            <p className="text-[15px] leading-[1.6] whitespace-pre-wrap flex-1">
              {slide.text}
            </p>

            {slide.visual_description && (
              <p className="text-xs italic text-muted-foreground">
                {slide.visual_description}
              </p>
            )}

            <span
              className={`text-xs tabular-nums self-end ${
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
        )}
      </div>

      <div className="flex items-center justify-center gap-4">
        <button
          type="button"
          onClick={goPrev}
          disabled={currentIndex === 0}
          className="inline-flex items-center justify-center rounded-md p-1.5 text-muted-foreground hover:text-foreground disabled:opacity-30 disabled:pointer-events-none transition-colors"
          aria-label="Previous slide"
        >
          <ChevronLeft className="h-5 w-5" />
        </button>
        <span className="text-sm tabular-nums text-muted-foreground min-w-[4rem] text-center">
          {currentIndex + 1} / {totalSlides}
        </span>
        <button
          type="button"
          onClick={goNext}
          disabled={currentIndex === totalSlides - 1}
          className="inline-flex items-center justify-center rounded-md p-1.5 text-muted-foreground hover:text-foreground disabled:opacity-30 disabled:pointer-events-none transition-colors"
          aria-label="Next slide"
        >
          <ChevronRight className="h-5 w-5" />
        </button>
      </div>

      {payload.hashtags.length > 0 && (
        <div className="flex flex-wrap justify-center gap-1.5">
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
  );
}
