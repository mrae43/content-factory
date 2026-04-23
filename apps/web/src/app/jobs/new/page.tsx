"use client";

import { useState } from "react";
import { useCreateJob } from "@/hooks/use-jobs";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function NewJobPage() {
  const router = useRouter();
  const createJob = useCreateJob();
  const [topic, setTopic] = useState("");
  const [rawText, setRawText] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const result = await createJob.mutateAsync({ topic, raw_text: rawText });
    router.push(`/jobs/${result.id}`);
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
            <Button type="submit" disabled={createJob.isPending}>
              {createJob.isPending ? "Creating..." : "Create Job"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
