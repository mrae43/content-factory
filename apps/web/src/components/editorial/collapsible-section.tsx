"use client";

import { useState } from "react";

interface CollapsibleSectionProps {
  label: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

export function CollapsibleSection({
  label,
  children,
  defaultOpen = false,
}: CollapsibleSectionProps) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="mt-3">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-1.5 py-2 text-xs font-medium tracking-wide text-muted-foreground hover:text-primary transition-colors cursor-pointer"
        aria-expanded={open}
      >
        <span className="text-[10px] leading-none">
          {open ? "\u25BC" : "\u25B6"}
        </span>
        {label}
      </button>
      {open && (
        <div className="mt-2 rounded-md bg-muted p-3 font-mono text-xs leading-relaxed whitespace-pre-wrap break-words">
          {children}
        </div>
      )}
    </div>
  );
}
