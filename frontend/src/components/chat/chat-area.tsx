"use client";

import React, { useRef, useEffect } from "react";
import { useChatStore, Message } from "@/store/chat-store";
import { Copy, Check, ExternalLink } from "lucide-react";
import Image from "next/image";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
} from "@/components/ui/tooltip";

function TypingIndicator() {
    return (
        <div className="flex items-start gap-4 animate-fade-in-up">
            <div className="flex-shrink-0 mt-1">
                <Image src="/frappe_icon.png" alt="Frappe" width={28} height={28} className="rounded-lg drop-shadow-sm object-contain" />
            </div>
            <div className="bg-transparent pt-3.5">
                <div className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-full bg-primary typing-dot" />
                    <div className="w-2 h-2 rounded-full bg-primary typing-dot" />
                    <div className="w-2 h-2 rounded-full bg-primary typing-dot" />
                </div>
            </div>
        </div>
    );
}

function SourceChips({ sources }: { sources: { title: string; url: string; snippet?: string }[] }) {
    if (!sources || sources.length === 0) return null;

    return (
        <div className="flex flex-wrap gap-2 mt-3">
            {sources.map((source, idx) => {
                const isWeb = source.url.startsWith("http");
                let domain = "";
                try {
                    domain = isWeb ? new URL(source.url).hostname.replace("www.", "") : "";
                } catch {
                    domain = source.url;
                }

                // RAG document source with snippet tooltip
                if (!isWeb && source.snippet) {
                    return (
                        <Tooltip key={idx}>
                            <TooltipTrigger asChild>
                                <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-primary/30 bg-primary/5 hover:bg-primary/10 text-[12px] font-medium text-primary/80 hover:text-primary transition-all duration-200 cursor-default group">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="opacity-70 group-hover:opacity-100"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /><line x1="16" y1="13" x2="8" y2="13" /><line x1="16" y1="17" x2="8" y2="17" /></svg>
                                    <span className="max-w-[160px] truncate">{source.title}</span>
                                </div>
                            </TooltipTrigger>
                            <TooltipContent
                                side="top"
                                className="max-w-[400px] p-3 text-[12px] leading-relaxed bg-popover border border-border/60 shadow-xl rounded-xl"
                            >
                                <p className="font-semibold text-foreground/90 mb-1.5 text-[11px] uppercase tracking-wider">{source.title}</p>
                                <p className="text-muted-foreground whitespace-pre-wrap">{source.snippet}</p>
                            </TooltipContent>
                        </Tooltip>
                    );
                }

                // RAG document source without snippet
                if (!isWeb) {
                    return (
                        <div
                            key={idx}
                            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-primary/30 bg-primary/5 text-[12px] font-medium text-primary/80 cursor-default"
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="opacity-70"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /><line x1="16" y1="13" x2="8" y2="13" /><line x1="16" y1="17" x2="8" y2="17" /></svg>
                            <span className="max-w-[160px] truncate">{source.title}</span>
                        </div>
                    );
                }

                // Web source
                return (
                    <a
                        key={idx}
                        href={source.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-border/60 bg-card/60 hover:bg-accent hover:border-primary/30 text-[12px] font-medium text-foreground/70 hover:text-foreground transition-all duration-200 group"
                    >
                        <img
                            src={`https://www.google.com/s2/favicons?domain=${domain}&sz=16`}
                            alt=""
                            width={14}
                            height={14}
                            className="rounded-sm opacity-70 group-hover:opacity-100"
                        />
                        <span className="max-w-[160px] truncate">{source.title || domain}</span>
                        <ExternalLink size={10} className="opacity-0 group-hover:opacity-60 transition-opacity" />
                    </a>
                );
            })}
        </div>
    );
}

function MessageBubble({ message }: { message: Message }) {
    const [copied, setCopied] = React.useState(false);
    const isUser = message.role === "user";

    const handleCopy = async () => {
        await navigator.clipboard.writeText(message.content);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className="group relative flex items-start gap-3 sm:gap-4 animate-fade-in-up md:px-4">
            {/* Avatar */}
            <div className="flex-shrink-0 mt-1.5">
                {isUser ? (
                    <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-gradient-to-br from-primary/80 to-primary text-primary-foreground flex items-center justify-center font-semibold text-xs drop-shadow-sm">
                        U
                    </div>
                ) : (
                    <Image src="/frappe_icon.png" alt="Frappe" width={28} height={28} className="rounded-lg drop-shadow-sm object-contain" />
                )}
            </div>

            {/* Message Content */}
            <div className="flex-1 min-w-0 pt-1">
                <div className="font-semibold text-[13px] text-foreground/80 mb-1 tracking-tight">
                    {isUser ? "Sen" : "Frappe"}
                </div>

                {isUser ? (
                    <div className="text-[15px] leading-relaxed text-foreground whitespace-pre-wrap break-words">
                        {message.content}
                    </div>
                ) : (
                    <div className="prose prose-sm dark:prose-invert max-w-none text-[15px] leading-relaxed text-foreground break-words
                        prose-p:my-1.5 prose-p:leading-relaxed
                        prose-headings:mt-4 prose-headings:mb-2 prose-headings:font-semibold
                        prose-ul:my-2 prose-ol:my-2 prose-li:my-0.5
                        prose-code:text-[13px] prose-code:bg-muted prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md prose-code:font-mono prose-code:before:content-none prose-code:after:content-none
                        prose-pre:bg-zinc-900 prose-pre:rounded-xl prose-pre:border prose-pre:border-border/40
                        prose-a:text-primary prose-a:no-underline hover:prose-a:underline prose-a:font-medium
                        prose-strong:text-foreground prose-strong:font-semibold
                        prose-blockquote:border-l-primary/50 prose-blockquote:bg-muted/30 prose-blockquote:rounded-r-lg prose-blockquote:py-1 prose-blockquote:px-4
                    ">
                        <ReactMarkdown remarkPlugins={[remarkGfm]} components={{
                            a: ({ href, children, ...props }) => (
                                <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
                                    {children}
                                </a>
                            ),
                        }}>
                            {message.content}
                        </ReactMarkdown>
                    </div>
                )}

                {/* Source Chips */}
                {!isUser && message.sources && <SourceChips sources={message.sources} />}

                {/* Actions */}
                {!isUser && (
                    <div className="flex items-center gap-2 mt-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <button
                                    onClick={handleCopy}
                                    className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
                                >
                                    {copied ? <Check size={14} /> : <Copy size={14} />}
                                </button>
                            </TooltipTrigger>
                            <TooltipContent>{copied ? "Kopyalandı!" : "Kopyala"}</TooltipContent>
                        </Tooltip>
                    </div>
                )}
            </div>
        </div>
    );
}

export function ChatArea() {
    const { conversations, activeConversationId, isGenerating } = useChatStore();
    const bottomRef = useRef<HTMLDivElement>(null);

    const activeConversation = conversations.find(
        (c) => c.id === activeConversationId
    );

    useEffect(() => {
        if (bottomRef.current) {
            bottomRef.current.scrollIntoView({ behavior: "smooth" });
        }
    }, [activeConversation?.messages, isGenerating]);

    if (!activeConversation) return null;

    return (
        <div className="max-w-3xl mx-auto py-8 px-2 space-y-8">
            {activeConversation.messages.map((message) => (
                <MessageBubble key={message.id} message={message} />
            ))}
            {isGenerating && <TypingIndicator />}
            <div ref={bottomRef} className="h-2" />
        </div>
    );
}
