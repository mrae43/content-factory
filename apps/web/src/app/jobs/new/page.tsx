"use client";

import { useState } from "react";
import { useCreateJob } from "@/hooks/use-jobs";
import { useRouter } from "next/navigation";
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

const FORMAT_OPTIONS = [
  { value: "all", label: "All Formats" },
  { value: "video", label: "Video" },
  { value: "blog", label: "Blog" },
  { value: "carousel", label: "Carousel" },
] as const;

const PLATFORM_OPTIONS = [
  { value: "_none", label: "None" },
  { value: "twitter", label: "Twitter / X" },
  { value: "linkedin", label: "LinkedIn" },
  { value: "instagram", label: "Instagram" },
  { value: "youtube", label: "YouTube" },
] as const;

export default function NewJobPage() {
  const router = useRouter();
  const createJob = useCreateJob();
  const [topic, setTopic] = useState("");
  const [rawText, setRawText] = useState("");
  const [formatType, setFormatType] = useState("all");
  const [platform, setPlatform] = useState("_none");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      setErrorMessage(null);
      const result = await createJob.mutateAsync({
        topic,
        pre_context: {
          raw_text: rawText,
          source_urls: [],
          target_audience: "General",
          guardrail_strictness: "High",
        },
        strict_compliance_mode: true,
        format_type: formatType as "all" | "video" | "blog" | "carousel",
        platform:
          platform === "_none"
            ? undefined
            : (platform as "twitter" | "linkedin" | "instagram" | "youtube"),
      });
      router.push(`/jobs/${result.id}`);
    } catch (err) {
      setErrorMessage(
        err instanceof Error ? err.message : "Please try again."
      );
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <Card>
        <CardHeader>
          <CardTitle>Create New Job</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
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
            {errorMessage && (
              <Alert variant="destructive">
                <AlertTitle>Couldn&apos;t create job</AlertTitle>
                <AlertDescription>{errorMessage}</AlertDescription>
              </Alert>
            )}
            <Button type="submit" disabled={createJob.isPending}>
              {createJob.isPending ? "Creating..." : "Create Job"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
