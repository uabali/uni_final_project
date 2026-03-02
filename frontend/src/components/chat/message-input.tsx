"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { useChatStore } from "@/store/chat-store";
import { Button } from "@/components/ui/button";
import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
} from "@/components/ui/tooltip";
import {
    Paperclip,
    Mic,
    Globe,
    Square,
    ArrowUp,
    X,
    FileText,
    BookOpen,
    Cpu,
} from "lucide-react";
import { chatStream, uploadFiles, connectTaskStream, setDepartmentHeader } from "@/lib/api";
import { useSettingsStore } from "@/store/settings-store";
import { useTaskStore } from "@/store/task-store";

interface MessageInputProps {
    externalValue?: string;
    onExternalValueConsumed?: () => void;
}

const MCP_PRESETS = [
    {
        id: "filesystem",
        label: "Dosya Çalışma Alanı",
        description: "/data altındaki dosyaları oku, yaz ve özetle.",
        suggestedPrompt:
            "MCP dosya sistemi araçlarını kullanarak /data klasöründeki önemli dokümanları listele ve hangi dosyalarla başlayabileceğimi öner.",
    },
    {
        id: "postgres",
        label: "Postgres Analiz",
        description: "Okuma amaçlı SQL sorguları ile veri analizi.",
        suggestedPrompt:
            "MCP Postgres aracını kullanarak son 50 satırlık log tablosunu incele ve hata oranlarını özetle.",
    },
    {
        id: "memory",
        label: "Agent Hafıza Arama",
        description: "Önceki sohbet ve notları hafızadan tara.",
        suggestedPrompt:
            "MCP memory_search aracını kullanarak benimle ilgili kayıtlı önemli notları bul ve kısa bir özet çıkar.",
    },
];

export function MessageInput({
    externalValue,
    onExternalValueConsumed,
}: MessageInputProps) {
    const [input, setInput] = useState("");
    const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
    const [uploadedFileNames, setUploadedFileNames] = useState<string[]>([]);
    const [isUploading, setIsUploading] = useState(false);
    const [useRag, setUseRag] = useState(false);
    const [webSearchOnly, setWebSearchOnly] = useState(false);
    const [useMcp, setUseMcp] = useState(false);
    const [selectedMcpPreset, setSelectedMcpPreset] = useState<string | null>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const {
        activeConversationId,
        createConversation,
        addMessage,
        addMessageWithId,
        appendMessageContent,
        setMessageSources,
        setMessageMeta,
        isGenerating,
        setIsGenerating,
    } = useChatStore();
    const settings = useSettingsStore();
    const { upsertTask, clearTask } = useTaskStore();

    useEffect(() => {
        if (externalValue) {
            setInput(externalValue);
            onExternalValueConsumed?.();
            setTimeout(() => {
                textareaRef.current?.focus();
                adjustHeight();
            }, 50);
        }
    }, [externalValue, onExternalValueConsumed]);

    const adjustHeight = () => {
        const textarea = textareaRef.current;
        if (textarea) {
            textarea.style.height = "auto";
            textarea.style.height = Math.min(textarea.scrollHeight, 200) + "px";
        }
    };

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (files) {
            setAttachedFiles((prev) => [...prev, ...Array.from(files)]);
        }
        // Reset so same file can be selected again
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    };

    const removeFile = (index: number) => {
        setAttachedFiles((prev) => prev.filter((_, i) => i !== index));
    };

    const streamResponse = useCallback(
        async (conversationId: string, userMessage: string) => {
            // Department bilgisini header'a yansıt
            if (settings.departmentId) {
                setDepartmentHeader(settings.departmentId);
            }

            setIsGenerating(true);
            const assistantMessageId = Math.random().toString(36).substring(2, 15) + Date.now().toString(36);

            // Initial empty message to show UI bubble immediately
            addMessageWithId(conversationId, {
                id: assistantMessageId,
                role: "assistant",
                content: "",
            });

            let disconnectTaskStream: (() => void) | null = null;

            try {
                // Task status WebSocket'ini aynı session / conversation ID ile aç
                disconnectTaskStream = connectTaskStream(conversationId, (status) => {
                    upsertTask(status);
                });

                await chatStream(userMessage, conversationId, (event) => {
                    if (event.type === "token") {
                        appendMessageContent(conversationId, assistantMessageId, event.data.text);
                    } else if (event.type === "sources") {
                        // Parse sources and set them on the message
                        const sources = Array.isArray(event.data)
                            ? event.data.map((s: any) => ({
                                title: s.title || s.source || "",
                                url: s.url || s.source || "",
                                snippet: s.snippet || "",
                            }))
                            : [];
                        if (sources.length > 0) {
                            setMessageSources(conversationId, assistantMessageId, sources);
                        }
                    } else if (event.type === "done") {
                        // Capture latency and token count from done event
                        const latency = event.data?.latency_ms;
                        const tokens = event.data?.token_count;
                        setMessageMeta(conversationId, assistantMessageId, {
                            latency_ms: latency,
                            token_count: tokens,
                        });
                    } else if (event.type === "error") {
                        appendMessageContent(conversationId, assistantMessageId, `\n\n**Hata:** ${event.data.error}`);
                    }
                }, useRag);
            } catch (error) {
                console.error("Stream failed:", error);
                appendMessageContent(conversationId, assistantMessageId, "\n\n**Bağlantı Hatası:** API'ye erişilemiyor. Lütfen backend'in açık olduğundan emin olun.");
            } finally {
                if (disconnectTaskStream) {
                    disconnectTaskStream();
                }
                setIsGenerating(false);
            }
        },
        [addMessageWithId, appendMessageContent, setMessageSources, setMessageMeta, setIsGenerating, useRag, settings.departmentId, upsertTask]
    );

    const handleSend = useCallback(async () => {
        const message = input.trim();
        if (!message || isGenerating || isUploading) return;

        let convId = activeConversationId;
        if (!convId) {
            convId = createConversation();
        }

        // Upload attached files first (if any)
        let appendedFilesText = "";
        if (attachedFiles.length > 0) {
            setIsUploading(true);
            try {
                await uploadFiles(attachedFiles);
                const newNames = attachedFiles.map((f) => f.name);
                appendedFilesText = `\n\n[Sisteme yeni yüklenen dosyalar: ${newNames.join(", ")}]`;
                // Track uploaded files so user knows they don't need to re-upload
                setUploadedFileNames((prev) => [...new Set([...prev, ...newNames])]);
                setAttachedFiles([]);
            } catch (error) {
                console.error("File upload failed:", error);
                // Still send the message even if upload fails
            } finally {
                setIsUploading(false);
            }
        }

        const finalMessage = message + appendedFilesText;
        const mcpTag = useMcp
            ? `[MCP${selectedMcpPreset ? `:${selectedMcpPreset}` : ""}] `
            : "";
        const backendMessage = webSearchOnly
            ? `[WEB_ONLY] ${mcpTag}${finalMessage}`
            : `${mcpTag}${finalMessage}`;

        addMessage(convId, {
            role: "user",
            content: finalMessage,
        });

        setInput("");
        if (textareaRef.current) {
            textareaRef.current.style.height = "auto";
        }

        streamResponse(convId, backendMessage);
    }, [
        input,
        isGenerating,
        isUploading,
        activeConversationId,
        createConversation,
        addMessage,
        attachedFiles,
        streamResponse,
        webSearchOnly,
        useMcp,
        selectedMcpPreset,
    ]);

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleStop = () => {
        setIsGenerating(false);
    };

    return (
        <div className="w-full max-w-3xl mx-auto px-4 pb-6 pt-2">
            <div
                className="
          relative flex flex-col gap-1 
          bg-card/95 backdrop-blur-xl border border-border/80 
          rounded-[26px] px-3 py-2.5 
          shadow-[0_4_24px_rgba(0,0,0,0.06)] dark:shadow-[0_4_24px_rgba(0,0,0,0.4)]
          transition-all duration-300 focus-within:ring-2 focus-within:ring-primary/20 focus-within:border-primary/40
        "
            >
                {/* Mode toggles inside the input card */}
                <div className="flex items-center gap-2 px-2 pt-1 pb-0.5">
                    <span className="text-[11px] font-medium text-muted-foreground/70">
                        Mod:
                    </span>
                    <button
                        type="button"
                        onClick={() => {
                            const next = !webSearchOnly;
                            setWebSearchOnly(next);
                            // Web arama modu açıkken RAG'in de etkin olması gerekiyor (web_search tool'u için)
                            if (next && !useRag) {
                                setUseRag(true);
                            }
                        }}
                        className={`flex items-center gap-1.5 text-[11px] font-medium px-2.5 py-1 rounded-full border transition-all duration-200 ${
                            webSearchOnly
                                ? "bg-primary/10 border-primary/40 text-primary hover:bg-primary/15"
                                : "border-border/50 text-foreground/60 hover:text-foreground hover:bg-accent"
                        }`}
                    >
                        <Globe size={13} />
                        <span>İnternette Ara</span>
                        {webSearchOnly && (
                            <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                        )}
                    </button>
                    <button
                        type="button"
                        onClick={() => {
                            const next = !useRag;
                            setUseRag(next);
                        }}
                        className={`flex items-center gap-1.5 text-[11px] font-medium px-2.5 py-1 rounded-full border transition-all duration-200 ${
                            useRag
                                ? "bg-primary/10 border-primary/40 text-primary hover:bg-primary/15"
                                : "border-border/50 text-foreground/60 hover:text-foreground hover:bg-accent"
                        }`}
                    >
                        <BookOpen size={13} />
                        <span>RAG</span>
                        {useRag && (
                            <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                        )}
                    </button>
                    <button
                        type="button"
                        onClick={() => {
                            const next = !useMcp;
                            setUseMcp(next);
                            if (!next) {
                                setSelectedMcpPreset(null);
                            }
                        }}
                        className={`flex items-center gap-1.5 text-[11px] font-medium px-2.5 py-1 rounded-full border transition-all duration-200 ${
                            useMcp
                                ? "bg-primary/10 border-primary/40 text-primary hover:bg-primary/15"
                                : "border-border/50 text-foreground/60 hover:text-foreground hover:bg-accent"
                        }`}
                    >
                        <Cpu size={13} />
                        <span>MCP</span>
                        {useMcp && (
                            <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                        )}
                    </button>
                </div>
                {/* Attached Files */}
                {attachedFiles.length > 0 && (
                    <div className="flex flex-wrap gap-4 px-3 pt-3 pb-2">
                        {attachedFiles.map((file, index) => {
                            const extension = file.name.split('.').pop()?.toUpperCase() || 'FILE';
                            return (
                                <div
                                    key={`${file.name}-${index}`}
                                    className="relative flex items-center gap-3 bg-transparent border border-border/60 rounded-[18px] p-2 min-w-[200px] max-w-[280px] group animate-fade-in-up"
                                >
                                    <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-[#d9534f] text-white flex-shrink-0 shadow-sm">
                                        <FileText size={22} strokeWidth={2} />
                                    </div>
                                    <div className="flex flex-col flex-grow min-w-0 pr-4">
                                        <span className="text-[14px] font-semibold truncate text-foreground/90">{file.name}</span>
                                        <span className="text-[12px] text-muted-foreground font-medium">{extension}</span>
                                    </div>
                                    <button
                                        onClick={() => removeFile(index)}
                                        className="absolute top-[-6px] right-[-6px] flex items-center justify-center w-[22px] h-[22px] rounded-full bg-foreground text-background shadow-md hover:scale-105 transition-transform"
                                    >
                                        <X size={12} strokeWidth={3} />
                                    </button>
                                </div>
                            );
                        })}
                    </div>
                )}

                <div className="flex items-end gap-2 px-1">
                    {/* Hidden file input */}
                    <input
                        ref={fileInputRef}
                        type="file"
                        multiple
                        className="hidden"
                        onChange={handleFileSelect}
                        accept=".pdf,.txt,.md,.doc,.docx,.csv,.json,.py,.js,.ts,.html,.css"
                    />

                    <Tooltip>
                        <TooltipTrigger asChild>
                            <button
                                type="button"
                                className="flex-shrink-0 p-2 rounded-full text-foreground/60 hover:bg-accent hover:text-foreground transition-colors mb-0.5"
                                disabled={isGenerating || isUploading}
                                onClick={() => fileInputRef.current?.click()}
                            >
                                <Paperclip size={20} strokeWidth={1.75} />
                            </button>
                        </TooltipTrigger>
                        <TooltipContent>Dosya Ekle</TooltipContent>
                    </Tooltip>

                    <textarea
                        ref={textareaRef}
                        className="
              flex-1 bg-transparent border-0 outline-none resize-none
              text-[15.5px] text-foreground placeholder:text-muted-foreground/70
              max-h-[200px] min-h-[24px] py-2.5 leading-relaxed
            "
                        placeholder="Frappe'ye mesaj gönder..."
                        value={input}
                        onChange={(e) => {
                            setInput(e.target.value);
                            adjustHeight();
                        }}
                        onKeyDown={handleKeyDown}
                        rows={1}
                        disabled={isGenerating || isUploading}
                    />

                    <div className="flex items-center gap-1.5 mb-0.5">
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <button
                                    type="button"
                                    className="flex-shrink-0 p-2 rounded-full text-foreground/60 hover:bg-accent hover:text-foreground transition-colors"
                                    disabled={isGenerating}
                                >
                                    <Mic size={20} strokeWidth={1.75} />
                                </button>
                            </TooltipTrigger>
                            <TooltipContent>Sesli Giriş (Yakında)</TooltipContent>
                        </Tooltip>

                        {isGenerating ? (
                            <Button
                                size="icon"
                                className="h-[34px] w-[34px] rounded-full bg-foreground text-background hover:bg-foreground/90 shadow-sm"
                                onClick={handleStop}
                            >
                                <Square size={13} fill="currentColor" />
                            </Button>
                        ) : (
                            <Button
                                size="icon"
                                className={`
                  h-[34px] w-[34px] rounded-full transition-all duration-200 shadow-sm
                  ${input.trim()
                                        ? "bg-foreground text-background hover:bg-foreground/80 scale-100"
                                        : "bg-[#e5e5e5] dark:bg-[#2c2c2c] text-[#a0a0a0] dark:text-[#555555] cursor-not-allowed"
                                    }
                `}
                                onClick={handleSend}
                                disabled={!input.trim() || isUploading}
                            >
                                <ArrowUp size={18} strokeWidth={2.5} />
                            </Button>
                        )}
                    </div>
                </div>
            </div>

            {/* Uploaded files indicator */}
            {uploadedFileNames.length > 0 && (
                <div className="flex items-center justify-center gap-2 mt-2 text-[11px] text-muted-foreground/70 font-medium">
                    <FileText size={12} className="text-primary/60" />
                    <span>Yüklü dosyalar: {uploadedFileNames.join(", ")}</span>
                    <span className="text-muted-foreground/40">— tekrar yüklemenize gerek yok</span>
                    <button
                        onClick={() => setUploadedFileNames([])}
                        className="ml-1 p-0.5 rounded-full hover:bg-accent transition-colors"
                    >
                        <X size={10} />
                    </button>
                </div>
            )}

            {useMcp && (
                <div className="mt-2 flex flex-wrap gap-2 px-2">
                    {MCP_PRESETS.map((preset) => (
                        <button
                            key={preset.id}
                            type="button"
                            onClick={() => {
                                setSelectedMcpPreset(preset.id);
                                setInput(preset.suggestedPrompt);
                                setTimeout(() => {
                                    adjustHeight();
                                    textareaRef.current?.focus();
                                }, 10);
                            }}
                            className={`flex items-start gap-2 px-3 py-2 rounded-2xl border text-left transition-all duration-200 text-[11px] ${
                                selectedMcpPreset === preset.id
                                    ? "bg-primary/10 border-primary/40 text-primary-foreground/90 dark:text-primary-foreground"
                                    : "border-border/50 text-foreground/70 hover:bg-accent"
                            }`}
                        >
                            <Cpu size={14} className="mt-0.5 text-primary" />
                            <div className="flex flex-col">
                                <span className="font-semibold text-[11px]">
                                    {preset.label}
                                </span>
                                <span className="text-[10px] text-muted-foreground/80">
                                    {preset.description}
                                </span>
                            </div>
                        </button>
                    ))}
                </div>
            )}

            <p className="text-[11px] text-center text-muted-foreground/60 mt-3 font-medium tracking-wide">
                Yapay zeka hata yapabilir. Lütfen önemli bilgileri doğrulayın.
            </p>
        </div>
    );
}
