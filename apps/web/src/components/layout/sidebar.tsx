"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useUIStore } from "@/stores/ui-store";
import { cn } from "@/lib/utils";
import { useEffect } from "react";

const navItems = [
  { href: "/", label: "Overview" },
  { href: "/jobs", label: "Stories" },
  { href: "/jobs/new", label: "Commission" },
];

function Masthead() {
  return (
    <div className="px-5 pt-6 pb-4">
      <h2 className="font-heading text-[1.25rem] font-bold leading-tight tracking-[-0.01em] text-foreground">
        Content Factory
      </h2>
      <div className="mt-1.5 h-[2px] w-[4.5rem] bg-primary" />
    </div>
  );
}

function DarkModeToggle() {
  const { theme, setTheme } = useUIStore();

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
  }, [theme]);

  return (
    <button
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
      className="flex items-center gap-2 px-5 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
      aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
    >
      {theme === "dark" ? (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="5" />
          <line x1="12" y1="1" x2="12" y2="3" />
          <line x1="12" y1="21" x2="12" y2="23" />
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
          <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
          <line x1="1" y1="12" x2="3" y2="12" />
          <line x1="21" y1="12" x2="23" y2="12" />
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
          <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
        </svg>
      ) : (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
        </svg>
      )}
    </button>
  );
}

export function Sidebar() {
  const pathname = usePathname();
  const { sidebarOpen, toggleSidebar } = useUIStore();

  return (
    <>
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 md:hidden"
          onClick={toggleSidebar}
          aria-hidden="true"
        />
      )}
      <aside
        className={cn(
          "flex flex-col border-r border-border bg-background",
          "fixed inset-y-0 left-0 z-50 w-60 shrink-0 transition-transform duration-200 md:static md:z-auto md:transition-[width,opacity]",
          !sidebarOpen && "-translate-x-full md:translate-x-0 md:w-0 md:overflow-hidden md:border-r-0 md:opacity-0"
        )}
      >
        <Masthead />
        <nav className="mt-2 flex flex-col gap-1 px-3">
          {navItems.map((item) => {
            const isActive = (() => {
              if (item.href === "/") return pathname === "/";
              if (item.href === "/jobs/new") return pathname === "/jobs/new";
              return pathname === "/jobs" || (pathname.startsWith("/jobs/") && pathname !== "/jobs/new");
            })();

            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => {
                  if (window.innerWidth < 768) toggleSidebar();
                }}
                className={cn(
                  "block rounded-none py-2 pl-[17px] pr-4 text-[0.875rem] font-medium leading-none transition-colors",
                  isActive
                    ? "border-l-[3px] border-primary bg-accent text-primary"
                    : "border-l-[3px] border-transparent text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                )}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="mt-auto border-t border-border px-3 pb-4 pt-3">
          <DarkModeToggle />
          <p className="px-2 pt-2 text-[0.6875rem] font-medium tracking-[0.02em] text-muted-foreground/60">
            v1.0
          </p>
        </div>
      </aside>
    </>
  );
}
