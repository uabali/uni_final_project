"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { useChatStore } from "@/store/chat-store";
import { Button } from "@/components/ui/button";
import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
} from "@/components/ui/tooltip";
import { Paperclip, Mic, Globe, Square, ArrowUp, X, FileText, BookOpen } from "lucide-react";
import { chatStream, uploadFiles } from "@/lib/api";

interface MessageInputProps {
    externalValue?: string;
    onExternalValueConsumed?: () => void;
}

export function MessageInput({
    externalValue,
    onExternalValueConsumed,
}: MessageInputProps) {
    const [input, setInput] = useState("");
    const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
    const [isUploading, setIsUploading] = useState(false);
    const [useRag, setUseRag] = useState(false);
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const {
        activeConversationId,
        createConversation,
        addMessage,
        addMessageWithId,
        appendMessageContent,
        setMessageSources,
        isGenerating,
        setIsGenerating,
    } = useChatStore();

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
            setIsGenerating(true);
            const assistantMessageId = Math.random().toString(36).substring(2, 15) + Date.now().toString(36);

            // Initial empty message to show UI bubble immediately
            addMessageWithId(conversationId, {
                id: assistantMessageId,
                role: "assistant",
                content: "",
            });

            try {
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
                    } else if (event.type === "error") {
                        appendMessageContent(conversationId, assistantMessageId, `\n\n**Hata:** ${event.data.error}`);
                    }
                }, useRag);
            } catch (error) {
                console.error("Stream failed:", error);
                appendMessageContent(conversationId, assistantMessageId, "\n\n**Bağlantı Hatası:** API'ye erişilemiyor. Lütfen backend'in açık olduğundan emin olun.");
            } finally {
                setIsGenerating(false);
            }
        },
        [addMessageWithId, appendMessageContent, setMessageSources, setIsGenerating, useRag]
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
                appendedFilesText = `\n\n[Sisteme yeni yüklenen dosyalar: ${attachedFiles.map(f => f.name).join(", ")}]`;
                setAttachedFiles([]);
            } catch (error) {
                console.error("File upload failed:", error);
                // Still send the message even if upload fails
            } finally {
                setIsUploading(false);
            }
        }

        const finalMessage = message + appendedFilesText;

        addMessage(convId, {
            role: "user",
            content: finalMessage,
        });

        setInput("");
        if (textareaRef.current) {
            textareaRef.current.style.height = "auto";
        }

        streamResponse(convId, finalMessage);
    }, [
        input,
        isGenerating,
        isUploading,
        activeConversationId,
        createConversation,
        addMessage,
        attachedFiles,
        streamResponse,
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
                {/* Attached Files */}
                {attachedFiles.length > 0 && (
                    <div className="flex flex-wrap gap-2 px-2 pt-1 pb-1">
                        {attachedFiles.map((file, index) => (
                            <div
                                key={`${file.name}-${index}`}
                                className="flex items-center gap-1.5 bg-muted/50 border border-border/60 rounded-lg px-2.5 py-1.5 text-[12px] font-medium text-foreground/80 group/chip animate-fade-in-up"
                            >
                                <FileText size={14} className="text-primary/70 flex-shrink-0" />
                                <span className="max-w-[120px] truncate">{file.name}</span>
                                <button
                                    onClick={() => removeFile(index)}
                                    className="ml-0.5 p-0.5 rounded-full text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
                                >
                                    <X size={12} />
                                </button>
                            </div>
                        ))}
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

                <div className="flex items-center gap-2 justify-between px-2 pt-0.5 pb-1">
                    <div className="flex items-center gap-2">
                        <button className="flex items-center gap-1.5 text-[11px] font-medium text-foreground/60 hover:text-foreground px-2.5 py-1 rounded-full border border-border/50 hover:bg-accent transition-colors">
                            <Globe size={13} />
                            <span>İnternette Ara</span>
                        </button>
                        <button
                            type="button"
                            onClick={() => setUseRag(!useRag)}
                            className={`flex items-center gap-1.5 text-[11px] font-medium px-2.5 py-1 rounded-full border transition-all duration-200 ${useRag
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
                    </div>
                </div>
            </div>

            <p className="text-[11px] text-center text-muted-foreground/60 mt-3 font-medium tracking-wide">
                Yapay zeka hata yapabilir. Lütfen önemli bilgileri doğrulayın.
            </p>
        </div>
    );
}
