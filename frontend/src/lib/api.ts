export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/** Backend API key (X-API-Key) — settings'ten set edilir. */
let backendApiKey = "";

export function setBackendApiKey(key: string) {
    backendApiKey = key;
}

function getBaseHeaders(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (backendApiKey) h["X-API-Key"] = backendApiKey;
    return h;
}

export interface ChatResponse {
    session_id: string;
    answer: string;
    latency_ms: number;
    used_provider: string;
    sources: any[];
}

export interface StreamEvent {
    type: "token" | "sources" | "done" | "error";
    data: any;
}

export async function chatStream(
    message: string,
    sessionId?: string,
    onEvent?: (event: StreamEvent) => void,
    useRag: boolean = true
): Promise<void> {
    const headers = getBaseHeaders();
    if (sessionId) headers["X-Session-ID"] = sessionId;

    try {
        const response = await fetch(`${API_URL}/chat`, {
            method: "POST",
            headers,
            body: JSON.stringify({ message, use_rag: useRag }),
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.statusText}`);
        }

        const reader = response.body?.getReader();
        const decoder = new TextDecoder("utf-8");

        if (!reader) {
            throw new Error("No reader available from response");
        }

        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            const parts = buffer.split("\n\n");
            // Keep last part in buffer if it doesn't end with double newline
            buffer = parts.pop() || "";

            for (const part of parts) {
                if (!part.trim()) continue;

                const eventMatch = part.match(/event:\s*([^\n]+)/);
                const dataMatch = part.match(/data:\s*([^\n]+)/);

                if (eventMatch && dataMatch && onEvent) {
                    const type = eventMatch[1].trim() as StreamEvent["type"];
                    try {
                        const data = JSON.parse(dataMatch[1].trim());
                        onEvent({ type, data });
                    } catch (e) {
                        console.error("Error parsing JSON from stream:", dataMatch[1]);
                    }
                }
            }
        }
    } catch (error) {
        let errorMessage = (error as Error).message;
        if (errorMessage.includes("Failed to fetch")) {
            errorMessage = "Backend sunucusuna ulaşılamıyor. Lütfen (localhost:8000) FastAPI servisinin açık olduğundan emin olun.";
        }
        if (onEvent) {
            onEvent({ type: "error", data: { error: errorMessage } });
        }
        throw new Error(errorMessage);
    }
}

export async function setLlmConfig(provider: string, model?: string, apiKey?: string, baseUrl?: string) {
    const body: Record<string, string> = { provider };
    if (model) body.model = model;
    if (apiKey) body.api_key = apiKey;
    if (baseUrl) body.base_url = baseUrl;

    try {
        const response = await fetch(`${API_URL}/config/llm`, {
            method: "PUT",
            headers: getBaseHeaders(),
            body: JSON.stringify(body),
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Model ayarı güncellenemedi: ${errorText}`);
        }
        return response.json();
    } catch (error) {
        if (error instanceof TypeError && (error as Error).message === "Failed to fetch") {
            throw new Error(
                `Backend sunucusuna ulaşılamıyor. Lütfen FastAPI servisini başlatın: uvicorn api.app:app --host 0.0.0.0 --port 8000`
            );
        }
        throw error;
    }
}

export async function uploadFiles(files: File[]): Promise<{ status: string; ingested: number; skipped: number; errors: string[] }> {
    const formData = new FormData();
    files.forEach((file) => {
        formData.append("files", file);
    });

    const headers: Record<string, string> = {};
    if (backendApiKey) headers["X-API-Key"] = backendApiKey;

    const response = await fetch(`${API_URL}/ingest/upload`, {
        method: "POST",
        headers,
        body: formData,
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Dosya yükleme hatası: ${errorText}`);
    }
    return response.json();
}
