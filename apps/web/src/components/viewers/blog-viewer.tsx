"use client";

import type { BlogFormatPayload } from "@content-factory/shared-types";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface BlogViewerProps {
  payload: BlogFormatPayload;
}

export function BlogViewer({ payload }: BlogViewerProps) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">{payload.title}</h1>
        {payload.subtitle && (
          <p className="mt-1 text-muted-foreground text-lg">{payload.subtitle}</p>
        )}
      </div>

      {payload.sections.map((section, i) => (
        <Card key={i}>
          <CardContent className="pt-6 space-y-3">
            <h2 className="text-lg font-semibold">{section.heading}</h2>
            <div className="whitespace-pre-wrap text-sm">{section.body}</div>
            {section.key_takeaway && (
              <div className="rounded-md bg-muted p-3 text-sm">
                <span className="font-medium">Key Takeaway:</span>{" "}
                {section.key_takeaway}
              </div>
            )}
            {section.sources_used && section.sources_used.length > 0 && (
              <p className="text-xs text-muted-foreground">
                Sources: {section.sources_used.join(", ")}
              </p>
            )}
          </CardContent>
        </Card>
      ))}

      <details className="rounded-lg border p-4">
        <summary className="cursor-pointer text-sm font-medium">
          SEO Metadata
        </summary>
        <div className="mt-3 space-y-1 text-sm">
          <p>
            <span className="font-medium">Title:</span>{" "}
            {payload.seo_meta.meta_title}
          </p>
          <p>
            <span className="font-medium">Description:</span>{" "}
            {payload.seo_meta.meta_description}
          </p>
          <div className="flex flex-wrap gap-1 mt-1">
            {payload.seo_meta.keywords.map((kw) => (
              <Badge key={kw} variant="outline" className="text-xs">
                {kw}
              </Badge>
            ))}
          </div>
          {payload.seo_meta.canonical_url && (
            <p>
              <span className="font-medium">Canonical URL:</span>{" "}
              {payload.seo_meta.canonical_url}
            </p>
          )}
        </div>
      </details>

      {payload.tags.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {payload.tags.map((tag) => (
            <Badge key={tag} variant="secondary">
              {tag}
            </Badge>
          ))}
        </div>
      )}

      {payload.call_to_action && (
        <Card>
          <CardContent className="pt-4">
            <p className="text-sm font-medium">Call to Action</p>
            <p className="text-sm text-muted-foreground">
              {payload.call_to_action}
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
