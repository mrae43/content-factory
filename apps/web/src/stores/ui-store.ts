import { create } from "zustand";

function detectTheme(): "light" | "dark" {
  if (typeof window === "undefined") return "dark";
  try {
    const stored = localStorage.getItem("theme");
    if (stored === "dark" || stored === "light") return stored;
  } catch {}
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

interface UIState {
  sidebarOpen: boolean;
  selectedJobFilter: string;
  theme: "light" | "dark";
  toggleSidebar: () => void;
  setJobFilter: (filter: string) => void;
  setTheme: (theme: "light" | "dark") => void;
}

export const useUIStore = create<UIState>((set) => ({
  sidebarOpen: false,
  selectedJobFilter: "all",
  theme: detectTheme(),
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  setJobFilter: (filter) => set({ selectedJobFilter: filter }),
  setTheme: (theme) => set({ theme }),
}));
