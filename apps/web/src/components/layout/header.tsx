"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

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

  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b border-border bg-card px-6">
      <h1 className="font-heading text-[1.125rem] font-semibold text-foreground">
        {title}
      </h1>
      {!isCommission && (
        <Link
          href="/jobs/new"
          className="inline-flex h-8 items-center rounded-md bg-primary px-4 text-[0.8125rem] font-medium text-primary-foreground transition-colors hover:bg-primary/90"
        >
          Commission
        </Link>
      )}
    </header>
  );
}
