import { create } from "zustand";

interface UIState {
  sidebarOpen: boolean;
  selectedJobFilter: string;
  theme: "light" | "dark" | "system";
  toggleSidebar: () => void;
  setJobFilter: (filter: string) => void;
  setTheme: (theme: "light" | "dark" | "system") => void;
}

export const useUIStore = create<UIState>((set) => ({
  sidebarOpen: true,
  selectedJobFilter: "all",
  theme: "system",
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  setJobFilter: (filter) => set({ selectedJobFilter: filter }),
  setTheme: (theme) => set({ theme }),
}));
