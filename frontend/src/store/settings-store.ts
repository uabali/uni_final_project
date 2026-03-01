import { create } from "zustand";
import { persist } from "zustand/middleware";

interface SettingsState {
    openaiKey: string;
    anthropicKey: string;
    ollamaUrl: string;
    openRouterKey: string;
    geminiKey: string;
    backendApiKey: string;

    // Actions
    setOpenaiKey: (key: string) => void;
    setAnthropicKey: (key: string) => void;
    setOllamaUrl: (url: string) => void;
    setOpenRouterKey: (key: string) => void;
    setGeminiKey: (key: string) => void;
    setBackendApiKey: (key: string) => void;
    clearKeys: () => void;
}

export const useSettingsStore = create<SettingsState>()(
    persist(
        (set) => ({
            openaiKey: "",
            anthropicKey: "",
            ollamaUrl: "",
            openRouterKey: "",
            geminiKey: "",
            backendApiKey: "",

            setOpenaiKey: (key) => set({ openaiKey: key }),
            setAnthropicKey: (key) => set({ anthropicKey: key }),
            setOllamaUrl: (url) => set({ ollamaUrl: url }),
            setOpenRouterKey: (key) => set({ openRouterKey: key }),
            setGeminiKey: (key) => set({ geminiKey: key }),
            setBackendApiKey: (key) => set({ backendApiKey: key }),

            clearKeys: () => set({
                openaiKey: "",
                anthropicKey: "",
                ollamaUrl: "",
                openRouterKey: "",
                geminiKey: "",
                backendApiKey: "",
            }),
        }),
        {
            name: "frappe-settings-storage", // name of the item in the storage (must be unique)
            // partialize allows picking only the state that needs to be persisted
            partialize: (state) => ({
                openaiKey: state.openaiKey,
                anthropicKey: state.anthropicKey,
                ollamaUrl: state.ollamaUrl,
                openRouterKey: state.openRouterKey,
                geminiKey: state.geminiKey,
                backendApiKey: state.backendApiKey
            }),
        }
    )
);
