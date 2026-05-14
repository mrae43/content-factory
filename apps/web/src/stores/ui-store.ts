import { create } from "zustand";

interface UIState {
  sidebarOpen: boolean;
  selectedJobFilter: string;
  theme: "light" | "dark";
  toggleSidebar: () => void;
  setJobFilter: (filter: string) => void;
  setTheme: (theme: "light" | "dark") => void;
}

export const useUIStore = create<UIState>((set) => ({
  sidebarOpen: true,
  selectedJobFilter: "all",
  theme: "light",
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  setJobFilter: (filter) => set({ selectedJobFilter: filter }),
  setTheme: (theme) => set({ theme }),
}));
