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

const FORMAT_OPTIONS = [
  { value: "All", label: "All Formats" },
  { value: "Video", label: "Video" },
  { value: "Blog", label: "Blog" },
  { value: "Carousel", label: "Carousel" },
] as const;

const PLATFORM_OPTIONS = [
  { value: "None", label: "None" },
  { value: "Twitter", label: "Twitter / X" },
  { value: "LinkedIn", label: "LinkedIn" },
  { value: "Instagram", label: "Instagram" },
  { value: "Youtube", label: "YouTube" },
] as const;

const STRICTNESS_OPTIONS = [
  { value: "Low", label: "Low" },
  { value: "Medium", label: "Medium" },
  { value: "High", label: "High" },
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
  const [formatType, setFormatType] = useState("All");
  const [platform, setPlatform] = useState("None");
  const [targetAudience, setTargetAudience] = useState("General");
  const [guardrailStrictness, setGuardrailStrictness] = useState("High");
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
        platform:
          platform === "none"
            ? undefined
            : (platform as "twitter" | "linkedin" | "instagram" | "youtube"),
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
    setShowConfirmation(true);
  };

  const formatLabel = FORMAT_OPTIONS.find((o) => o.value === formatType)?.label;
  const platformLabel = platform !== "none" ? PLATFORM_OPTIONS.find((o) => o.value === platform)?.label : null;

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
                    >
                      <SelectTrigger className="w-full h-11 text-base sm:h-9 sm:text-sm">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {FORMAT_OPTIONS.map((opt) => (
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
                      onValueChange={(v) => v !== null && setPlatform(v)}
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

              {showConfirmation ? (
                <div className="rounded-lg bg-muted p-4 sm:p-5 space-y-3">
                  <p className="font-heading text-[0.875rem] font-semibold text-foreground sm:text-[0.9375rem]">
                    Confirm your assignment
                  </p>
                  <div className="space-y-1.5">
                    <p className="text-sm text-foreground">
                      &ldquo;{topic}&rdquo;
                    </p>
                    <p className="text-[0.75rem] font-medium tracking-[0.02em] text-muted-foreground">
                      {formatLabel}
                      {platformLabel ? ` · ${platformLabel}` : ""}
                      {" · "}{guardrailStrictness} strictness
                      {rawText ? " · Brief provided" : ""}
                      {parsedSourceUrls.length > 0
                        ? ` · ${parsedSourceUrls.length} source${parsedSourceUrls.length > 1 ? "s" : ""}`
                        : ""}
                    </p>
                  </div>
                  <div className="flex flex-col-reverse gap-2 pt-1 sm:flex-row">
                    <Button
                      type="button"
                      disabled={createJob.isPending}
                      onClick={handleSubmit}
                      className="font-heading font-semibold h-11 sm:h-auto"
                    >
                      {createJob.isPending
                        ? "Commissioning..."
                        : "Confirm & Commission"}
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => setShowConfirmation(false)}
                      disabled={createJob.isPending}
                      className="h-11 sm:h-auto"
                    >
                      Edit
                    </Button>
                  </div>
                </div>
              ) : (
                <Button
                  type="submit"
                  disabled={createJob.isPending}
                  className="w-full font-heading text-[0.9375rem] font-semibold h-11 sm:h-10"
                >
                  {createJob.isPending
                    ? "Commissioning..."
                    : "Commission This Story"}
                </Button>
              )}
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
