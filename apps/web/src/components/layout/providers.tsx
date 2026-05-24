"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import { useState, useEffect, useLayoutEffect } from "react";
import { useUIStore } from "@/stores/ui-store";

const useIsomorphicLayoutEffect =
  typeof window !== "undefined" ? useLayoutEffect : useEffect;

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [queryClient] = useState(() => new QueryClient());

  useIsomorphicLayoutEffect(() => {
    const mq = window.matchMedia("(min-width: 768px)");
    useUIStore.setState({ sidebarOpen: mq.matches });

    function handleChange(e: MediaQueryListEvent) {
      useUIStore.setState({ sidebarOpen: e.matches });
    }

    mq.addEventListener("change", handleChange);

    const isDark = document.documentElement.classList.contains("dark");
    const currentTheme = useUIStore.getState().theme;
    if ((isDark && currentTheme !== "dark") || (!isDark && currentTheme !== "light")) {
      useUIStore.setState({ theme: isDark ? "dark" : "light" });
    }

    return () => mq.removeEventListener("change", handleChange);
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <div className="flex h-screen">
          <Sidebar />
          <div className="flex flex-1 flex-col overflow-hidden">
            <Header />
            <main id="main" className="flex-1 overflow-y-auto p-4 md:p-6">
              {children}
            </main>
          </div>
        </div>
      </TooltipProvider>
    </QueryClientProvider>
  );
}
