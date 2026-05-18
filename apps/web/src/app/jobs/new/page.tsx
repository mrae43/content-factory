"use client";

import { Suspense, useState } from "react";
import { useCreateJob } from "@/hooks/use-jobs";
import { useRouter, useSearchParams } from "next/navigation";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
} from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogOverlay,
  DialogPortal,
  DialogTitle,
} from "@/components/ui/dialog";

const FORMAT_OPTIONS = [
  { value: "all", label: "All Formats" },
  { value: "video", label: "Video" },
  { value: "blog", label: "Blog" },
  { value: "carousel", label: "Carousel" },
] as const;

const PLATFORM_FORMATS: Record<string, typeof FORMAT_OPTIONS[number]["value"][]> = {
  twitter: ["all", "blog", "carousel", "video"],
  linkedin: ["all", "carousel", "blog"],
  instagram: ["all", "carousel", "video"],
  tiktok: ["all", "carousel", "video"],
  youtube: ["all", "blog", "video"],
};

const PLATFORM_OPTIONS = [
  { value: "twitter", label: "Twitter / X" },
  { value: "linkedin", label: "LinkedIn" },
  { value: "instagram", label: "Instagram" },
  { value: "tiktok", label: "TikTok" },
  { value: "youtube", label: "YouTube" },
] as const;

const STRICTNESS_OPTIONS = [
  { value: "low", label: "Low" },
  { value: "medium", label: "Medium" },
  { value: "high", label: "High" },
] as const;

export default function NewJobPage() {
  return (
    <Suspense>
      <NewJobForm />
    </Suspense>
  );
}

function NewJobForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const createJob = useCreateJob();
  const [topic, setTopic] = useState(() => searchParams.get("topic") ?? "");
  const [rawText, setRawText] = useState(
    () => searchParams.get("raw_text") ?? ""
  );
  const [sourceUrls, setSourceUrls] = useState("");
  const [formatType, setFormatType] = useState("all");
  const [platform, setPlatform] = useState("");
  const [targetAudience, setTargetAudience] = useState("general");
  const [guardrailStrictness, setGuardrailStrictness] = useState("high");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [researchOpen, setResearchOpen] = useState(false);

  const parsedSourceUrls = sourceUrls
    .split("\n")
    .map((u) => u.trim())
    .filter(Boolean);

  const handleSubmit = async () => {
    try {
      setErrorMessage(null);
      const result = await createJob.mutateAsync({
        topic,
        pre_context: {
          raw_text: rawText || undefined,
          source_urls: parsedSourceUrls.length > 0 ? parsedSourceUrls : [],
          target_audience: targetAudience,
          guardrail_strictness: guardrailStrictness,
        },
        strict_compliance_mode: true,
        format_type: formatType as "all" | "video" | "blog" | "carousel",
        platform: platform as "twitter" | "linkedin" | "instagram" | "youtube" | "tiktok",
      });
      router.push(`/jobs/${result.id}`);
      toast.success("Story commissioned — pipeline started.");
    } catch (err) {
      setErrorMessage(
        err instanceof Error ? err.message : "Please try again."
      );
      setShowConfirmation(false);
    }
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!platform) return;
    setShowConfirmation(true);
  };

  const availableFormats = platform
    ? FORMAT_OPTIONS.filter((o) => PLATFORM_FORMATS[platform]?.includes(o.value))
    : [];

  const formatLabel = FORMAT_OPTIONS.find((o) => o.value === formatType)?.label;
  const platformLabel = PLATFORM_OPTIONS.find((o) => o.value === platform)?.label;

  return (
    <div className="px-4 py-6 sm:px-6 sm:py-8 md:py-10 lg:py-12">
      <div className="mx-auto max-w-2xl">
        <form onSubmit={handleFormSubmit}>
          <div className="rounded-lg border border-border bg-card shadow-[0_1px_2px_rgba(31,28,24,0.04)] overflow-hidden">
            <div className="px-4 pt-5 pb-2 sm:px-6 sm:pt-6">
              <h1 className="font-heading text-[1.25rem] font-semibold tracking-[-0.01em] text-foreground sm:text-[1.5rem]">
                Commission Content
              </h1>
              <p className="mt-1 text-[0.75rem] font-medium tracking-[0.02em] text-muted-foreground">
                Give the newsroom its assignment.
              </p>
            </div>

            <div className="px-4 pb-6 space-y-6 sm:px-6 sm:pb-8 sm:space-y-8">
              <div className="space-y-2">
                <label
                  htmlFor="headline"
                  className="block font-heading text-[1rem] font-semibold text-foreground sm:text-[1.125rem]"
                >
                  Headline
                </label>
                <Input
                  id="headline"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  placeholder="What's the story?"
                  required
                  className="h-11 text-base sm:h-9 sm:text-sm"
                />
              </div>

              <div className="space-y-2">
                <label
                  htmlFor="brief"
                  className="block font-heading text-[1rem] font-semibold text-foreground sm:text-[1.125rem]"
                >
                  Editorial Brief
                </label>
                <Textarea
                  id="brief"
                  value={rawText}
                  onChange={(e) => setRawText(e.target.value)}
                  placeholder="Background, key points, angle, tone..."
                  rows={5}
                  className="min-h-[100px] text-base sm:min-h-[120px] sm:text-sm sm:rows-6"
                />
              </div>

              <div className="space-y-3">
                <span className="block font-heading text-[1rem] font-semibold text-foreground sm:text-[1.125rem]">
                  Publication Target
                </span>
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                  <div className="space-y-1.5">
                    <span className="block text-[0.75rem] font-medium tracking-[0.02em] text-muted-foreground">
                      Format
                    </span>
                    <Select
                      value={formatType}
                      onValueChange={(v) => v !== null && setFormatType(v)}
                      disabled={!platform}
                    >
                      <SelectTrigger className="w-full h-11 text-base sm:h-9 sm:text-sm">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {availableFormats.map((opt) => (
                          <SelectItem key={opt.value} value={opt.value}>
                            {opt.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-1.5">
                    <span className="block text-[0.75rem] font-medium tracking-[0.02em] text-muted-foreground">
                      Platform
                    </span>
                    <Select
                      value={platform}
                      onValueChange={(v) => {
                        setPlatform(v ?? "");
                        setFormatType("all");
                      }}
                    >
                      <SelectTrigger className="w-full h-11 text-base sm:h-9 sm:text-sm">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {PLATFORM_OPTIONS.map((opt) => (
                          <SelectItem key={opt.value} value={opt.value}>
                            {opt.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-1.5">
                    <span className="block text-[0.75rem] font-medium tracking-[0.02em] text-muted-foreground">
                      Strictness
                    </span>
                    <Select
                      value={guardrailStrictness}
                      onValueChange={(v) => v !== null && setGuardrailStrictness(v)}
                    >
                      <SelectTrigger className="w-full h-11 text-base sm:h-9 sm:text-sm">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {STRICTNESS_OPTIONS.map((opt) => (
                          <SelectItem key={opt.value} value={opt.value}>
                            {opt.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>

              <Collapsible open={researchOpen} onOpenChange={setResearchOpen}>
                <CollapsibleTrigger
                  className={
                    "flex w-full items-center gap-1.5 py-1 text-[0.75rem] font-medium text-muted-foreground hover:text-primary transition-colors " +
                    (researchOpen ? "text-primary" : "")
                  }
                >
                  <span className="inline-block text-[0.625rem] transition-transform duration-0">
                    {researchOpen ? "\u25BC" : "\u25B6"}
                  </span>
                  <span className="font-heading text-[0.8125rem] font-semibold sm:text-[0.875rem]">
                    Research Materials
                  </span>
                </CollapsibleTrigger>
                <CollapsibleContent className="space-y-4 pt-4 sm:space-y-5">
                  <div className="space-y-2">
                    <span className="block text-[0.75rem] font-medium tracking-[0.02em] text-muted-foreground">
                      Source URLs
                    </span>
                    <Textarea
                      value={sourceUrls}
                      onChange={(e) => setSourceUrls(e.target.value)}
                      placeholder="https://..."
                      rows={3}
                      className="min-h-[60px] font-mono text-base sm:text-[0.8125rem]"
                    />
                  </div>
                  <div className="space-y-2">
                    <span className="block text-[0.75rem] font-medium tracking-[0.02em] text-muted-foreground">
                      Reference Text
                    </span>
                    <Textarea
                      value={rawText}
                      onChange={(e) => setRawText(e.target.value)}
                      placeholder="Raw text, book excerpts, reports..."
                      rows={4}
                      className="min-h-[80px] text-base sm:text-sm"
                    />
                  </div>
                  <div className="space-y-2">
                    <span className="block text-[0.75rem] font-medium tracking-[0.02em] text-muted-foreground">
                      Target Audience
                    </span>
                    <Input
                      value={targetAudience}
                      onChange={(e) => setTargetAudience(e.target.value)}
                      placeholder="e.g., Marketing professionals"
                      className="h-11 text-base sm:h-9 sm:text-sm"
                    />
                  </div>
                </CollapsibleContent>
              </Collapsible>

              {errorMessage && (
                <div className="rounded-lg border border-primary/20 bg-primary/5 p-3 sm:p-4 space-y-1">
                  <p className="text-sm font-medium text-primary">
                    Couldn&apos;t commission this story
                  </p>
                  <p className="text-sm text-muted-foreground">{errorMessage}</p>
                </div>
              )}

              <Button
                type="submit"
                disabled={createJob.isPending}
                className="w-full font-heading text-[0.9375rem] font-semibold h-11 sm:h-10"
              >
                {createJob.isPending
                  ? "Commissioning..."
                  : "Commission This Story"}
              </Button>
            </div>
          </div>
        </form>

        <Dialog open={showConfirmation} onOpenChange={setShowConfirmation}>
          <DialogPortal>
            <DialogOverlay className="bg-foreground/50 sm:bg-foreground/30 sm:backdrop-blur-sm" />
            <DialogContent
              showCloseButton={false}
              className="max-w-[calc(100%-1.5rem)] sm:max-w-md rounded-xl sm:rounded-lg border-border bg-card p-0 ring-0 shadow-[0_8px_32px_rgba(31,28,24,0.18)] sm:shadow-[0_4px_24px_rgba(31,28,24,0.12)]"
            >
              <DialogHeader className="gap-1.5 sm:gap-2 px-5 pt-5 pb-0 sm:px-6 sm:pt-6">
                <DialogTitle className="font-heading text-[1.125rem] sm:text-[1.25rem] font-semibold tracking-[-0.01em] text-foreground">
                  Confirm your assignment
                </DialogTitle>
                <DialogDescription className="text-[0.8125rem] sm:text-[0.75rem] font-medium tracking-[0.02em] text-muted-foreground">
                  Review before sending to the newsroom.
                </DialogDescription>
              </DialogHeader>

              <div className="px-5 py-4 sm:px-6 sm:py-5">
                <div className="rounded-lg bg-muted p-4 sm:p-4 space-y-2.5 sm:space-y-2">
                  <p className="font-heading text-[1rem] sm:text-[0.9375rem] font-semibold text-foreground leading-snug">
                    &ldquo;{topic}&rdquo;
                  </p>
                  <p className="text-[0.8125rem] sm:text-[0.75rem] font-medium tracking-[0.02em] text-muted-foreground">
                    {formatLabel}
                    {platformLabel ? ` · ${platformLabel}` : ""}
                    {guardrailStrictness ? ` · ${guardrailStrictness} strictness` : ""}
                    {rawText ? " · Brief provided" : ""}
                    {parsedSourceUrls.length > 0
                      ? ` · ${parsedSourceUrls.length} source${parsedSourceUrls.length > 1 ? "s" : ""}`
                      : ""}
                  </p>
                </div>
              </div>

              <DialogFooter className="mx-0 mb-0 border-t border-border bg-transparent px-5 py-4 sm:px-6 sm:py-4 gap-3">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setShowConfirmation(false)}
                  disabled={createJob.isPending}
                  className="h-11 sm:h-9 flex-1 sm:flex-none sm:min-w-[100px]"
                >
                  Edit
                </Button>
                <Button
                  type="button"
                  disabled={createJob.isPending}
                  onClick={handleSubmit}
                  className="font-heading font-semibold h-11 sm:h-9 flex-1 sm:flex-none sm:min-w-[180px]"
                >
                  {createJob.isPending
                    ? "Commissioning..."
                    : "Confirm & Commission"}
                </Button>
              </DialogFooter>
            </DialogContent>
          </DialogPortal>
        </Dialog>
      </div>
    </div>
  );
}
