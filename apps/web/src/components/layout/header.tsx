"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useUIStore } from "@/stores/ui-store";
import { StatusBar } from "./status-bar";

const pageTitles: Record<string, string> = {
  "/": "Overview",
  "/jobs": "Stories",
  "/jobs/new": "Commission Content",
};

function getPageTitle(pathname: string): string {
  if (pathname.startsWith("/jobs/")) {
    const slug = pathname.split("/jobs/")[1];
    if (slug === "new") return "Commission Content";
    if (slug) return "Story";
  }
  return pageTitles[pathname] ?? "Content Factory";
}

export function Header() {
  const pathname = usePathname();
  const title = getPageTitle(pathname);
  const isCommission = pathname === "/jobs/new";
  const { toggleSidebar } = useUIStore();

  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b border-border bg-card px-4 md:px-6">
      <div className="flex items-center gap-3">
        <button
          onClick={toggleSidebar}
          className="inline-flex h-8 w-8 items-center justify-center text-muted-foreground transition-colors hover:text-foreground md:hidden"
          aria-label="Toggle navigation"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <line x1="3" y1="6" x2="21" y2="6" />
            <line x1="3" y1="12" x2="21" y2="12" />
            <line x1="3" y1="18" x2="21" y2="18" />
          </svg>
        </button>
        <h1 className="font-heading text-[1.125rem] font-semibold text-foreground">
          {title}
        </h1>
      </div>
      <div className="flex items-center gap-3">
        <StatusBar />
        {!isCommission && (
          <Link
            href="/jobs/new"
            className="inline-flex h-8 items-center rounded-md bg-primary px-4 text-[0.8125rem] font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            Commission
          </Link>
        )}
      </div>
    </header>
  );
}
