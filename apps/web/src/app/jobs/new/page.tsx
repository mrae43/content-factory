"use client";

import { Suspense, useState, useEffect } from "react";
import { useCreateJob } from "@/hooks/use-jobs";
import { useRouter, useSearchParams } from "next/navigation";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
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
  { value: "all", label: "All Formats" },
  { value: "video", label: "Video" },
  { value: "blog", label: "Blog" },
  { value: "carousel", label: "Carousel" },
] as const;

const PLATFORM_OPTIONS = [
  { value: "none", label: "None" },
  { value: "twitter", label: "Twitter / X" },
  { value: "linkedin", label: "LinkedIn" },
  { value: "instagram", label: "Instagram" },
  { value: "youtube", label: "YouTube" },
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
  const [topic, setTopic] = useState("");
  const [rawText, setRawText] = useState("");

  useEffect(() => {
    const topicParam = searchParams.get("topic");
    const rawTextParam = searchParams.get("raw_text");
    if (topicParam) setTopic(topicParam);
    if (rawTextParam) setRawText(rawTextParam);
  }, [searchParams]);
  const [formatType, setFormatType] = useState("all");
  const [platform, setPlatform] = useState("none");
  const [targetAudience, setTargetAudience] = useState("General");
  const [guardrailStrictness, setGuardrailStrictness] = useState("High");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [showConfirmation, setShowConfirmation] = useState(false);

  const handleSubmit = async () => {
    try {
      setErrorMessage(null);
      const result = await createJob.mutateAsync({
        topic,
        pre_context: {
          raw_text: rawText,
          source_urls: [],
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
      toast.success("Job created — watching pipeline...");
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

  return (
    <div className="max-w-2xl mx-auto">
      <Card>
        <CardHeader>
          <CardTitle>Create New Job</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleFormSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="topic">Topic</Label>
              <Input
                id="topic"
                value={topic}
                onChange={(e) => setTopic(e.target.value)}
                placeholder="Enter video topic..."
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="raw_text">Raw Text / Brief</Label>
              <Textarea
                id="raw_text"
                value={rawText}
                onChange={(e) => setRawText(e.target.value)}
                placeholder="Paste your content brief, article, or source material..."
                rows={10}
                required
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Format Type</Label>
                <Select
                  value={formatType}
                  onValueChange={(v) => v !== null && setFormatType(v)}
                >
                  <SelectTrigger className="w-full">
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
              <div className="space-y-2">
                <Label>Platform (optional)</Label>
                <Select
                  value={platform}
                  onValueChange={(v) => v !== null && setPlatform(v)}
                >
                  <SelectTrigger className="w-full">
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
            </div>

            <Collapsible>
              <CollapsibleTrigger
                render={
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-muted-foreground"
                  />
                }
              >
                Advanced options
              </CollapsibleTrigger>
              <CollapsibleContent className="space-y-4 pt-2">
                <div className="space-y-2">
                  <Label>Target Audience</Label>
                  <Input
                    value={targetAudience}
                    onChange={(e) => setTargetAudience(e.target.value)}
                    placeholder="e.g., Marketing professionals"
                  />
                </div>
                <div className="space-y-2">
                  <Label>Guardrail Strictness</Label>
                  <Select
                    value={guardrailStrictness}
                    onValueChange={(v) =>
                      v !== null && setGuardrailStrictness(v)
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Low">Low</SelectItem>
                      <SelectItem value="Medium">Medium</SelectItem>
                      <SelectItem value="High">High</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CollapsibleContent>
            </Collapsible>

            {errorMessage && (
              <Alert variant="destructive">
                <AlertTitle>Couldn&apos;t create job</AlertTitle>
                <AlertDescription>{errorMessage}</AlertDescription>
              </Alert>
            )}

            {showConfirmation ? (
              <div className="bg-muted p-4 rounded-lg space-y-2">
                <p className="text-sm font-medium">Confirm your job:</p>
                <p className="text-sm">Topic: &ldquo;{topic}&rdquo;</p>
                <p className="text-sm text-muted-foreground">
                  Format:{" "}
                  {FORMAT_OPTIONS.find((o) => o.value === formatType)?.label}
                  {platform !== "none" &&
                    ` | Platform: ${PLATFORM_OPTIONS.find((o) => o.value === platform)?.label}`}
                </p>
                <div className="flex gap-2 pt-2">
                  <Button
                    type="button"
                    disabled={createJob.isPending}
                    onClick={handleSubmit}
                  >
                    {createJob.isPending ? "Creating..." : "Confirm & Create"}
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => setShowConfirmation(false)}
                    disabled={createJob.isPending}
                  >
                    Edit
                  </Button>
                </div>
              </div>
            ) : (
              <Button type="submit" disabled={createJob.isPending}>
                {createJob.isPending ? "Creating..." : "Create Job"}
              </Button>
            )}
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
