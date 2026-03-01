"use client";

import React, { useState, useEffect } from "react";
import { useSettingsStore } from "@/store/settings-store";
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
    Settings,
    Bot,
    Sparkles,
    Cpu,
    Key,
    GlobeLock,
    MessageSquareText,
    PanelLeftClose,
    CheckCircle2,
    ShieldCheck,
} from "lucide-react";
import Image from "next/image";

interface SettingsDialogProps {
    children: React.ReactNode;
}

export function SettingsDialog({ children }: SettingsDialogProps) {
    const [isOpen, setIsOpen] = useState(false);
    const [activeTab, setActiveTab] = useState<"general" | "api_keys">("api_keys");
    const [isMounted, setIsMounted] = useState(false);

    // Settings state from local storage
    const settings = useSettingsStore();

    // Local state for the inputs before saving
    const [localOpenai, setLocalOpenai] = useState("");
    const [localAnthropic, setLocalAnthropic] = useState("");
    const [localOllama, setLocalOllama] = useState("");
    const [localOpenRouter, setLocalOpenRouter] = useState("");
    const [localGemini, setLocalGemini] = useState("");
    const [localBackendApiKey, setLocalBackendApiKey] = useState("");

    const [isSavedHovered, setIsSavedHovered] = useState(false);
    const [savedStatus, setSavedStatus] = useState(false);

    useEffect(() => {
        setIsMounted(true);
    }, []);

    // Sync local state when dialog opens
    useEffect(() => {
        if (isOpen) {
            setLocalOpenai(settings.openaiKey);
            setLocalAnthropic(settings.anthropicKey);
            setLocalOllama(settings.ollamaUrl);
            setLocalOpenRouter(settings.openRouterKey);
            setLocalGemini(settings.geminiKey);
            setLocalBackendApiKey(settings.backendApiKey);
            setSavedStatus(false);
        }
    }, [
        isOpen,
        settings.openaiKey,
        settings.anthropicKey,
        settings.ollamaUrl,
        settings.openRouterKey,
        settings.geminiKey,
        settings.backendApiKey,
    ]);

    const handleSave = () => {
        settings.setOpenaiKey(localOpenai);
        settings.setAnthropicKey(localAnthropic);
        settings.setOllamaUrl(localOllama);
        settings.setOpenRouterKey(localOpenRouter);
        settings.setGeminiKey(localGemini);
        settings.setBackendApiKey(localBackendApiKey);

        setSavedStatus(true);
        setTimeout(() => {
            setSavedStatus(false);
            setIsOpen(false);
        }, 800);
    };

    if (!isMounted) return <>{children}</>;

    return (
        <Dialog open={isOpen} onOpenChange={setIsOpen}>
            <DialogTrigger asChild>
                {children}
            </DialogTrigger>
            <DialogContent className="max-w-3xl p-0 overflow-hidden border-border/40 shadow-2xl rounded-2xl bg-background/95 backdrop-blur-2xl">
                <div className="flex h-[550px]">
                    {/* Sidebar */}
                    <div className="w-[220px] bg-muted/30 border-r border-border/40 p-3 h-full flex flex-col gap-1">
                        <h2 className="px-3 py-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground/60 mb-2 mt-2">
                            Ayarlar
                        </h2>
                        <button
                            onClick={() => setActiveTab("general")}
                            className={`
                flex items-center gap-3 px-3 py-2.5 rounded-xl text-[14px] font-medium transition-all duration-200
                ${activeTab === "general"
                                    ? "bg-foreground/5 text-foreground"
                                    : "text-muted-foreground hover:bg-foreground/5 hover:text-foreground"
                                }
              `}
                        >
                            <Settings size={18} className={activeTab === "general" ? "text-primary" : "text-muted-foreground"} />
                            Genel
                        </button>
                        <button
                            onClick={() => setActiveTab("api_keys")}
                            className={`
                flex items-center gap-3 px-3 py-2.5 rounded-xl text-[14px] font-medium transition-all duration-200
                ${activeTab === "api_keys"
                                    ? "bg-foreground/5 text-foreground"
                                    : "text-muted-foreground hover:bg-foreground/5 hover:text-foreground"
                                }
              `}
                        >
                            <Key size={18} className={activeTab === "api_keys" ? "text-primary" : "text-muted-foreground"} />
                            API Bağlantıları
                        </button>

                        <div className="mt-auto px-3 py-4">
                            <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground/50">
                                <ShieldCheck size={14} />
                                <span>Şifreli yerel depolama</span>
                            </div>
                        </div>
                    </div>

                    {/* Content Area */}
                    <div className="flex-1 flex flex-col h-full bg-background relative">
                        <DialogHeader className="px-8 py-6 border-b border-border/40 pb-5">
                            <DialogTitle className="text-xl font-semibold tracking-tight">
                                {activeTab === "general" ? "Genel Ayarlar" : "API Bağlantıları"}
                            </DialogTitle>
                            <p className="text-[14px] text-muted-foreground mt-1.5">
                                {activeTab === "general"
                                    ? "Frappe asistanınızın temel yapılandırmasını özelleştirin."
                                    : "Kendi API anahtarlarınızı bağlayarak özel modelleri kullanın. Tüm veriler sadece tarayıcınızda (yerel) kriptolanarak saklanır."}
                            </p>
                        </DialogHeader>

                        <div className="flex-1 overflow-y-auto px-8 py-6">
                            {activeTab === "general" && (
                                <div className="space-y-6">
                                    <div className="flex flex-col gap-2">
                                        <label className="text-[13px] font-semibold tracking-wide text-muted-foreground uppercase flex items-center gap-2">
                                            <ShieldCheck size={15} />
                                            Backend API Anahtarı
                                        </label>
                                        <Input
                                            type="password"
                                            placeholder="API_KEY env ile korumali backend icin"
                                            value={localBackendApiKey}
                                            onChange={(e) => setLocalBackendApiKey(e.target.value)}
                                            className="flex h-10 w-full rounded-xl border border-border/80 bg-background/50 px-3 py-2 text-[14px] transition-colors focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary"
                                        />
                                        <p className="text-[12px] text-muted-foreground/80">Backend API_KEY env degiskeni tanimliysa bu alani doldurun.</p>
                                    </div>
                                    <div className="flex flex-col gap-2">
                                        <label className="text-[13px] font-semibold tracking-wide text-muted-foreground uppercase flex items-center gap-2">
                                            <GlobeLock size={15} />
                                            Vektör Veritabanı
                                        </label>
                                        <select className="flex h-10 w-full rounded-xl border border-border/80 bg-background/50 px-3 py-2 text-[14px] transition-colors focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary">
                                            <option>Qdrant (Local)</option>
                                            <option>Pinecone</option>
                                        </select>
                                    </div>
                                    <div className="flex flex-col gap-2">
                                        <label className="text-[13px] font-semibold tracking-wide text-muted-foreground uppercase flex items-center gap-2">
                                            <MessageSquareText size={15} />
                                            Sistem Yönergesi (System Prompt)
                                        </label>
                                        <textarea
                                            className="flex min-h-[120px] w-full rounded-xl border border-border/80 bg-background/50 px-3 py-2 text-[14px] leading-relaxed resize-none transition-colors focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary"
                                            placeholder="Sen Frappe, yardımcı bir asistansın..."
                                            defaultValue="Sen Frappe, alanında çok akıllı, yardımcı ve yaratıcı bir yapay zeka asistanısın. Kısa ve net cevaplar verirsin."
                                        />
                                    </div>
                                </div>
                            )}

                            {activeTab === "api_keys" && (
                                <div className="space-y-6">
                                    {/* OpenAI Key */}
                                    <div className="flex flex-col gap-2 relative group">
                                        <label className="text-[13px] font-bold tracking-wide text-foreground flex items-center justify-between">
                                            <span className="flex items-center gap-2">
                                                <Image src="/openai_icon.png" alt="OpenAI" width={16} height={16} className="object-contain" />
                                                OpenAI API Key
                                            </span>
                                            {localOpenai && <span className="text-[11px] font-medium text-emerald-500 bg-emerald-500/10 px-2 py-0.5 rounded-full">Bağlı</span>}
                                        </label>
                                        <Input
                                            type="password"
                                            placeholder="sk-proj-..."
                                            value={localOpenai}
                                            onChange={(e) => setLocalOpenai(e.target.value)}
                                            className="bg-transparent border-border/60 focus-visible:ring-1 focus-visible:ring-primary/50 focus-visible:border-primary/50 rounded-xl h-11 text-[14px]"
                                        />
                                    </div>

                                    {/* Anthropic / Claude */}
                                    <div className="flex flex-col gap-2 relative group">
                                        <label className="text-[13px] font-bold tracking-wide text-foreground flex items-center justify-between">
                                            <span className="flex items-center gap-2">
                                                <Image src="/claude_icon.png" alt="Claude" width={16} height={16} className="object-contain" />
                                                Anthropic (Claude) API Key
                                            </span>
                                            {localAnthropic && <span className="text-[11px] font-medium text-emerald-500 bg-emerald-500/10 px-2 py-0.5 rounded-full">Bağlı</span>}
                                        </label>
                                        <Input
                                            type="password"
                                            placeholder="sk-ant-..."
                                            value={localAnthropic}
                                            onChange={(e) => setLocalAnthropic(e.target.value)}
                                            className="bg-transparent border-border/60 focus-visible:ring-1 focus-visible:ring-primary/50 focus-visible:border-primary/50 rounded-xl h-11 text-[14px]"
                                        />
                                    </div>

                                    {/* OpenRouter */}
                                    <div className="flex flex-col gap-2 relative group">
                                        <label className="text-[13px] font-bold tracking-wide text-foreground flex items-center justify-between">
                                            <span className="flex items-center gap-2">
                                                <Image src="/openrouter-icon.webp" alt="OpenRouter" width={16} height={16} className="object-contain" />
                                                OpenRouter API Key
                                            </span>
                                            {localOpenRouter && <span className="text-[11px] font-medium text-emerald-500 bg-emerald-500/10 px-2 py-0.5 rounded-full">Bağlı</span>}
                                        </label>
                                        <Input
                                            type="password"
                                            placeholder="sk-or-v1-..."
                                            value={localOpenRouter}
                                            onChange={(e) => setLocalOpenRouter(e.target.value)}
                                            className="bg-transparent border-border/60 focus-visible:ring-1 focus-visible:ring-primary/50 focus-visible:border-primary/50 rounded-xl h-11 text-[14px]"
                                        />
                                    </div>

                                    {/* Gemini */}
                                    <div className="flex flex-col gap-2 relative group">
                                        <label className="text-[13px] font-bold tracking-wide text-foreground flex items-center justify-between">
                                            <span className="flex items-center gap-2">
                                                <Image src="/gemini_icon.webp" alt="Gemini" width={16} height={16} className="object-contain" />
                                                Google Gemini API Key
                                            </span>
                                            {localGemini && <span className="text-[11px] font-medium text-emerald-500 bg-emerald-500/10 px-2 py-0.5 rounded-full">Bağlı</span>}
                                        </label>
                                        <Input
                                            type="password"
                                            placeholder="AIzaSy..."
                                            value={localGemini}
                                            onChange={(e) => setLocalGemini(e.target.value)}
                                            className="bg-transparent border-border/60 focus-visible:ring-1 focus-visible:ring-primary/50 focus-visible:border-primary/50 rounded-xl h-11 text-[14px]"
                                        />
                                    </div>

                                    {/* Ollama Local URL */}
                                    <div className="flex flex-col gap-2 relative group pt-2 border-t border-border/30">
                                        <label className="text-[13px] font-bold tracking-wide text-foreground flex items-center justify-between">
                                            <span className="flex items-center gap-2">
                                                <Image src="/ollama-icon.webp" alt="Ollama" width={16} height={16} className="object-contain" />
                                                Ollama (Local) URL
                                            </span>
                                            {localOllama && <span className="text-[11px] font-medium text-emerald-500 bg-emerald-500/10 px-2 py-0.5 rounded-full">Bağlı</span>}
                                        </label>
                                        <Input
                                            type="text"
                                            placeholder="http://localhost:11434"
                                            value={localOllama}
                                            onChange={(e) => setLocalOllama(e.target.value)}
                                            className="bg-transparent border-border/60 focus-visible:ring-1 focus-visible:ring-primary/50 focus-visible:border-primary/50 rounded-xl h-11 text-[14px]"
                                        />
                                        <p className="text-[12px] text-muted-foreground/80 mt-1">Cihazınızda çalışan yerel modelleri bağlamak için varsayılan sunucu adresini girin.</p>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Footer Action */}
                        <div className="px-8 py-5 border-t border-border/40 bg-muted/10 flex justify-end">
                            <Button
                                onClick={handleSave}
                                className={`
                  rounded-full px-8 h-10 transition-all duration-300
                  ${savedStatus ? "bg-emerald-500 hover:bg-emerald-600 text-white" : ""}
                `}
                                onMouseEnter={() => setIsSavedHovered(true)}
                                onMouseLeave={() => setIsSavedHovered(false)}
                            >
                                {savedStatus ? (
                                    <span className="flex items-center gap-2 tracking-wide font-medium">
                                        <CheckCircle2 size={16} /> Kaydedildi
                                    </span>
                                ) : (
                                    <span className="tracking-wide font-medium">Değişiklikleri Kaydet</span>
                                )}
                            </Button>
                        </div>
                    </div>
                </div>
            </DialogContent>
        </Dialog>
    );
}
