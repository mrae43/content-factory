"use client";

import { useUIStore } from "@/stores/ui-store";
import { Button } from "@/components/ui/button";

export function Header() {
  const { toggleSidebar } = useUIStore();

  return (
    <header className="flex h-14 items-center gap-4 border-b bg-card px-6">
      <Button variant="ghost" size="sm" onClick={toggleSidebar}>
        Menu
      </Button>
      <h1 className="text-lg font-semibold">Content Factory</h1>
    </header>
  );
}
