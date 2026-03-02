export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/** Backend API key (X-API-Key) — settings'ten set edilir. */
let backendApiKey = "";

export function setBackendApiKey(key: string) {
    backendApiKey = key;
}

export function setDepartmentHeader(dept: string) {
    // Şu an için sadece runtime'da kullanılan header'ı etkilemek üzere
    // environment üzerinden değil, hafif bir global değişken olarak tutuyoruz.
    currentDepartmentId = dept;
}

let currentDepartmentId: string | undefined = undefined;

function getBaseHeaders(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (backendApiKey) h["X-API-Key"] = backendApiKey;
    if (currentDepartmentId) h["X-Department-ID"] = currentDepartmentId;
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

export interface TaskStatusEvent {
    task_id: string;
    status: string;
    progress?: number | null;
    last_error?: string | null;
    updated_at: string;
    [key: string]: any;
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

export function connectTaskStream(
    taskId: string,
    onStatus: (status: TaskStatusEvent) => void
): () => void {
    const protocol = API_URL.startsWith("https") ? "wss" : "ws";
    const url = API_URL.replace(/^https?/, protocol) + `/ws/tasks/${encodeURIComponent(taskId)}`;

    let socket: WebSocket | null = new WebSocket(url);

    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data) as TaskStatusEvent;
            onStatus(data);
        } catch (err) {
            console.error("Task WebSocket parse error:", err);
        }
    };

    socket.onerror = (event) => {
        console.error("Task WebSocket error:", event);
    };

    const cleanup = () => {
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.close();
        }
        socket = null;
    };

    return cleanup;
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

export async function deleteIngestedFiles(names: string[]): Promise<{ status: string; deleted: number; errors: string[] }> {
    const headers = getBaseHeaders();
    const response = await fetch(`${API_URL}/ingest/delete`, {
        method: "POST",
        headers,
        body: JSON.stringify({ paths: names }),
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Dosya silme hatası: ${errorText}`);
    }
    return response.json();
}

export async function getLlmConfig(): Promise<{ provider: string; model: string }> {
    try {
        const response = await fetch(`${API_URL}/config/llm`, {
            headers: getBaseHeaders(),
        });
        if (!response.ok) return { provider: "unknown", model: "unknown" };
        return response.json();
    } catch {
        return { provider: "unknown", model: "unknown" };
    }
}
