"use client";

import React, { useState, useEffect, useMemo } from "react";
import { useSettingsStore } from "@/store/settings-store";
import { useChatStore } from "@/store/chat-store";
import { API_URL } from "@/lib/api";
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

type SettingsTab = "general" | "api_keys" | "monitoring";

export function SettingsDialog({ children }: SettingsDialogProps) {
    const [isOpen, setIsOpen] = useState(false);
    const [activeTab, setActiveTab] = useState<SettingsTab>("api_keys");
    const [isMounted, setIsMounted] = useState(false);

    // Settings state from local storage
    const settings = useSettingsStore();
    const { conversations, activeConversationId } = useChatStore();

    const activeConversation = useMemo(
        () => conversations.find((c) => c.id === activeConversationId) || null,
        [conversations, activeConversationId],
    );

    const monitoringStats = useMemo(() => {
        if (!activeConversation) {
            return null;
        }
        const assistantMessages = activeConversation.messages.filter(
            (m) => m.role === "assistant",
        );
        if (assistantMessages.length === 0) {
            return null;
        }

        let totalTokens = 0;
        let totalLatency = 0;
        let latencyCount = 0;

        for (const msg of assistantMessages) {
            if (typeof msg.token_count === "number") {
                totalTokens += msg.token_count;
            }
            if (typeof msg.latency_ms === "number") {
                totalLatency += msg.latency_ms;
                latencyCount += 1;
            }
        }

        const avgLatency = latencyCount > 0 ? totalLatency / latencyCount : null;

        return {
            turns: assistantMessages.length,
            totalTokens,
            avgLatencyMs: avgLatency,
        };
    }, [activeConversation]);

    // Local state for the inputs before saving
    const [localOpenai, setLocalOpenai] = useState("");
    const [localAnthropic, setLocalAnthropic] = useState("");
    const [localOllama, setLocalOllama] = useState("");
    const [localOpenRouter, setLocalOpenRouter] = useState("");
    const [localGemini, setLocalGemini] = useState("");
    const [localBackendApiKey, setLocalBackendApiKey] = useState("");
    const [localDepartmentId, setLocalDepartmentId] = useState("engineering");
    const [metricsSummary, setMetricsSummary] = useState<any | null>(null);

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
            setLocalDepartmentId(settings.departmentId || "engineering");
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
        settings.setDepartmentId(localDepartmentId);

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
            <DialogContent className="max-w-4xl p-0 overflow-hidden border-border/40 shadow-2xl rounded-2xl bg-background/95 backdrop-blur-2xl">
                <div className="flex h-[700px] w-full max-w-[1100px]">
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
                        <button
                            onClick={() => setActiveTab("monitoring")}
                            className={`
                flex items-center gap-3 px-3 py-2.5 rounded-xl text-[14px] font-medium transition-all duration-200
                ${activeTab === "monitoring"
                                    ? "bg-foreground/5 text-foreground"
                                    : "text-muted-foreground hover:bg-foreground/5 hover:text-foreground"
                                }
              `}
                        >
                            <Cpu size={18} className={activeTab === "monitoring" ? "text-primary" : "text-muted-foreground"} />
                            Monitoring
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
                                {activeTab === "general"
                                    ? "Genel Ayarlar"
                                    : activeTab === "api_keys"
                                        ? "API Bağlantıları"
                                        : "Monitoring ve İzleme"}
                            </DialogTitle>
                            <p className="text-[14px] text-muted-foreground mt-1.5">
                                {activeTab === "general"
                                    ? "Frappe asistanınızın temel yapılandırmasını özelleştirin."
                                    : activeTab === "api_keys"
                                        ? "Kendi API anahtarlarınızı bağlayarak özel modelleri kullanın. Tüm veriler sadece tarayıcınızda (yerel) kriptolanarak saklanır."
                                        : "Sohbetlerinizin performansını ve LangSmith/LangChain tracing ile arka planda neler olduğunu izleyin."}
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
                                            <ShieldCheck size={15} />
                                            Varsayılan Departman
                                        </label>
                                        <select
                                            className="flex h-10 w-full rounded-xl border border-border/80 bg-background/50 px-3 py-2 text-[14px] transition-colors focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary"
                                            value={localDepartmentId}
                                            onChange={(e) => setLocalDepartmentId(e.target.value)}
                                        >
                                            <option value="engineering">Engineering</option>
                                            <option value="project_mgmt">Project Mgmt</option>
                                            <option value="hr">HR</option>
                                            <option value="finance">Finance</option>
                                        </select>
                                        <p className="text-[12px] text-muted-foreground/80">
                                            JWT entegrasyonu ile backend tarafında departman bazlı yetki ve RAG izolasyonu
                                            uygulanır. Local geliştirmede bu değer istek başlıklarında taşınır.
                                        </p>
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

                            {activeTab === "monitoring" && (
                                <div className="space-y-6">
                                    <div className="flex flex-col gap-2">
                                        <div className="flex items-center justify-between">
                                            <div className="flex items-center gap-2">
                                                <Cpu size={18} className="text-primary" />
                                                <span className="text-[13px] font-semibold tracking-wide text-muted-foreground uppercase">
                                                    Aktif Sohbet Özeti
                                                </span>
                                            </div>
                                        </div>
                                        {activeConversation && monitoringStats ? (
                                            <div className="mt-2 grid grid-cols-1 sm:grid-cols-3 gap-3">
                                                <div className="rounded-xl border border-border/60 bg-muted/30 px-3 py-2.5 flex flex-col gap-1">
                                                    <span className="text-[11px] text-muted-foreground/80 uppercase tracking-wide">
                                                        Tur Sayısı
                                                    </span>
                                                    <span className="text-[18px] font-semibold text-foreground">
                                                        {monitoringStats.turns}
                                                    </span>
                                                </div>
                                                <div className="rounded-xl border border-border/60 bg-muted/30 px-3 py-2.5 flex flex-col gap-1">
                                                    <span className="text-[11px] text-muted-foreground/80 uppercase tracking-wide">
                                                        Toplam Token
                                                    </span>
                                                    <span className="text-[18px] font-semibold text-foreground">
                                                        {monitoringStats.totalTokens || 0}
                                                    </span>
                                                    <span className="text-[11px] text-muted-foreground/70">
                                                        (Sadece backend'in raporladığı cevaplar)
                                                    </span>
                                                </div>
                                                <div className="rounded-xl border border-border/60 bg-muted/30 px-3 py-2.5 flex flex-col gap-1">
                                                    <span className="text-[11px] text-muted-foreground/80 uppercase tracking-wide">
                                                        Ortalama Gecikme
                                                    </span>
                                                    <span className="text-[18px] font-semibold text-foreground">
                                                        {monitoringStats.avgLatencyMs
                                                            ? `${(monitoringStats.avgLatencyMs / 1000).toFixed(1)} sn`
                                                            : "—"}
                                                    </span>
                                                </div>
                                            </div>
                                        ) : (
                                            <p className="text-[13px] text-muted-foreground/80">
                                                Şu anda aktif sohbet bulunmuyor ya da henüz yanıt üretilmedi. Token ve
                                                gecikme metrikleri ilk asistans cevabından sonra burada görünecek.
                                            </p>
                                        )}
                                    </div>

                                    <div className="flex flex-col gap-2">
                                        <span className="text-[13px] font-semibold tracking-wide text-muted-foreground uppercase flex items-center gap-2">
                                            LangSmith / LangChain Tracing
                                        </span>
                                        <p className="text-[13px] text-muted-foreground/85 leading-relaxed">
                                            Backend tarafında <code className="text-xs px-1 py-0.5 rounded bg-muted border border-border/60">LANGCHAIN_TRACING_V2=true</code>{" "}
                                            ve geçerli bir <code className="text-xs px-1 py-0.5 rounded bg-muted border border-border/60">LANGCHAIN_API_KEY</code> tanımlıysa,
                                            her <code className="text-xs px-1 py-0.5 rounded bg-muted border border-border/60">/chat</code> isteği LangSmith&apos;te{" "}
                                            <span className="font-semibold">local-ai-stack</span> projesi altında
                                            detaylı bir trace olarak görünür.
                                        </p>
                                        <ul className="list-disc list-inside text-[13px] text-muted-foreground/85 space-y-1.5">
                                            <li>Her node için run adı olarak <code className="text-xs px-1 py-0.5 rounded bg-muted border border-border/60">api_chat_stream</code>,{" "}
                                                <code className="text-xs px-1 py-0.5 rounded bg-muted border border-border/60">react_agent</code>, <code className="text-xs px-1 py-0.5 rounded bg-muted border border-border/60">search_documents</code>{" "}
                                                gibi isimler kullanılır.</li>
                                            <li>Token kullanımı ve gecikme süresi, LangSmith UI&apos;ında ayrıca
                                                izlenebilir; bu ekrandaki özet sadece aktif sohbet için hafif bir
                                                özet sağlar.</li>
                                            <li>Daha agresif optimizasyon için sistem prompt uzunluğunu ve RAG bağlam
                                                boyutunu çevresel değişkenler üzerinden düşürebilirsiniz
                                                (örneğin <code className="text-xs px-1 py-0.5 rounded bg-muted border border-border/60">SEARCH_TOOL_MAX_CHUNKS</code>,{" "}
                                                <code className="text-xs px-1 py-0.5 rounded bg-muted border border-border/60">SEARCH_TOOL_MAX_CHARS_PER_CHUNK</code>).</li>
                                        </ul>
                                    </div>

                                    <div className="flex flex-col gap-2">
                                        <span className="text-[13px] font-semibold tracking-wide text-muted-foreground uppercase flex items-center gap-2">
                                            Departman Bazlı Özet
                                        </span>
                                        <button
                                            className="self-start text-[11px] px-2 py-1 rounded-full border border-border/60 text-muted-foreground hover:text-foreground hover:bg-accent/40 transition-colors"
                                            onClick={async () => {
                                                try {
                                                    const res = await fetch(`${API_URL}/metrics/summary`);
                                                    if (!res.ok) {
                                                        throw new Error(await res.text());
                                                    }
                                                    const data = await res.json();
                                                    setMetricsSummary(data);
                                                } catch (e) {
                                                    console.error("Metrics fetch failed:", e);
                                                }
                                            }}
                                        >
                                            Son Özeti Yükle
                                        </button>
                                        {metricsSummary && (
                                            <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-3 text-[12px]">
                                                {metricsSummary.departments?.map((d: any) => (
                                                    <div
                                                        key={d.department_id}
                                                        className="rounded-xl border border-border/60 bg-muted/30 px-3 py-2 flex flex-col gap-1"
                                                    >
                                                        <span className="text-[11px] text-muted-foreground/80 uppercase tracking-wide">
                                                            {d.department_id}
                                                        </span>
                                                        <span className="text-[13px] font-semibold text-foreground">
                                                            {d.total_requests} istek
                                                        </span>
                                                        <span className="text-[11px] text-muted-foreground/80">
                                                            Ortalama gecikme:{" "}
                                                            {d.avg_latency_ms
                                                                ? `${(d.avg_latency_ms / 1000).toFixed(1)} sn`
                                                                : "—"}
                                                        </span>
                                                        <span className="text-[11px] text-muted-foreground/80">
                                                            Toplam token: {d.total_tokens}
                                                        </span>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
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
