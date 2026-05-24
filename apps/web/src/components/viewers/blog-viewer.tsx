"use client";

import type { BlogFormatPayload } from "@content-factory/shared-types";
import { CollapsibleSection } from "@/components/editorial/collapsible-section";
import { CopyButton } from "./copy-button";

interface BlogViewerProps {
  payload: BlogFormatPayload;
}

function blogToText(p: BlogFormatPayload): string {
  const parts: string[] = [];
  parts.push(`Title: ${p.title}`);
  if (p.subtitle) parts.push(`Subtitle: ${p.subtitle}`);
  parts.push("");
  for (const s of p.sections) {
    parts.push(s.heading);
    parts.push(s.body);
    parts.push("");
  }
  if (p.call_to_action) parts.push(`CTA: ${p.call_to_action}`);
  return parts.join("\n");
}

export function BlogViewer({ payload }: BlogViewerProps) {
  return (
    <div className="space-y-8">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="font-heading text-2xl font-bold tracking-tight">
            {payload.title}
          </h1>
          {payload.subtitle && (
            <p className="mt-1.5 font-heading text-base font-medium text-muted-foreground italic">
              {payload.subtitle}
            </p>
          )}
        </div>
        <CopyButton getContent={() => blogToText(payload)} label="Copy article" />
      </div>

      {payload.sections.map((section, i) => (
        <div key={i} className="space-y-3">
          <h2 className="font-heading text-xl font-semibold tracking-tight">
            {section.heading}
          </h2>
          <div
            className={`text-[15px] leading-[1.7] text-foreground ${
              i === 0 ? "drop-cap" : ""
            }`}
          >
            {section.body.split("\n").map((paragraph, pi) => (
              <p key={pi} className="mb-4 last:mb-0">
                {paragraph}
              </p>
            ))}
          </div>
          {section.key_takeaway && (
            <blockquote className="pull-quote my-4">
              {section.key_takeaway}
            </blockquote>
          )}
          <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
            {section.sources_used && section.sources_used.length > 0 ? (
              <p className="text-xs text-muted-foreground">
                Sources: {section.sources_used.length} research chunk
                {section.sources_used.length !== 1 ? "s" : ""}
              </p>
            ) : (
              <span />
            )}
            <p className="text-xs text-muted-foreground">
              {section.word_count} words
            </p>
          </div>
          {i < payload.sections.length - 1 && (
            <div className="editorial-rule pt-4" />
          )}
        </div>
      ))}

      <CollapsibleSection label="SEO Details">
        <div className="space-y-2">
          <p>
            <span className="font-semibold text-foreground">Meta Title:</span>{" "}
            <span className="text-foreground">{payload.seo_meta.meta_title}</span>
          </p>
          <p>
            <span className="font-semibold text-foreground">Meta Description:</span>{" "}
            <span className="text-foreground">{payload.seo_meta.meta_description}</span>
          </p>
          {payload.seo_meta.canonical_url && (
            <p>
              <span className="font-semibold text-foreground">Canonical URL:</span>{" "}
              <span className="text-foreground">{payload.seo_meta.canonical_url}</span>
            </p>
          )}
          {payload.seo_meta.keywords.length > 0 && (
            <div className="flex flex-wrap gap-1 pt-1">
              {payload.seo_meta.keywords.map((kw) => (
                <span
                  key={kw}
                  className="inline-flex items-center rounded-[4px] bg-primary/10 px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-[0.05em] text-primary"
                >
                  {kw}
                </span>
              ))}
            </div>
          )}
        </div>
      </CollapsibleSection>

      {payload.tags.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {payload.tags.map((tag) => (
            <span
              key={tag}
              className="text-sm text-muted-foreground font-heading italic"
            >
              {tag}
            </span>
          ))}
        </div>
      )}

      {payload.call_to_action && (
        <div className="rounded-md border border-primary/20 bg-primary/5 p-4">
          <p className="text-xs font-semibold uppercase tracking-wide text-primary mb-1">
            Call to Action
          </p>
          <p className="text-sm text-foreground">{payload.call_to_action}</p>
        </div>
      )}
    </div>
  );
}
