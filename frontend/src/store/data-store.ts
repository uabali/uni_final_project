import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface DataFileEntry {
    name: string;
    size: number;
    uploadedAt: string; // ISO string; görüntülemede Date'e çevrilebilir
}

interface DataState {
    files: DataFileEntry[];
    addFiles: (files: { name: string; size: number }[]) => void;
    removeFile: (name: string) => void;
    clear: () => void;
}

export const useDataStore = create<DataState>()(
    persist(
        (set) => ({
            files: [],
            addFiles: (files) =>
                set((state) => {
                    const now = new Date().toISOString();
                    const existingNames = new Set(state.files.map((f) => f.name));
                    const newEntries: DataFileEntry[] = [];
                    for (const f of files) {
                        if (existingNames.has(f.name)) continue;
                        newEntries.push({
                            name: f.name,
                            size: f.size,
                            uploadedAt: now,
                        });
                    }
                    return {
                        files: [...newEntries, ...state.files],
                    };
                }),
            removeFile: (name) =>
                set((state) => ({
                    files: state.files.filter((f) => f.name !== name),
                })),
            clear: () => set({ files: [] }),
        }),
        {
            name: "frappe-data-storage",
        }
    )
);

