"use client";

import React from "react";
import { useChatStore } from "@/store/chat-store";
import { Button } from "@/components/ui/button";
import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
} from "@/components/ui/tooltip";
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { PanelLeft, Settings, Sun, Moon, ChevronDown, Sparkles } from "lucide-react";
import { setLlmConfig } from "@/lib/api";
import { SettingsDialog } from "./settings-dialog";
import { useSettingsStore } from "@/store/settings-store";
import Image from "next/image";

export function ChatHeader() {
    const { isSidebarOpen, toggleSidebar, activeConversationId, conversations } =
        useChatStore();
    const settings = useSettingsStore();

    const [isDark, setIsDark] = React.useState(true);
    const [selectedModel, setSelectedModel] = React.useState("Frappe Pro");
    const [isChangingModel, setIsChangingModel] = React.useState(false);

    // Check if component mounted to avoid hydration mismatch
    const [isMounted, setIsMounted] = React.useState(false);
    React.useEffect(() => {
        setIsMounted(true);
    }, []);

    const activeConversation = conversations.find(
        (c) => c.id === activeConversationId
    );

    const handleToggleTheme = () => {
        setIsDark(!isDark);
        document.documentElement.classList.toggle("dark");
    };

    const handleModelChange = async (name: string, provider: string, modelStr?: string, apiKey?: string, baseUrl?: string) => {
        if (selectedModel === name) return;
        setIsChangingModel(true);
        try {
            await setLlmConfig(provider, modelStr, apiKey, baseUrl);
            setSelectedModel(name);
        } catch (error) {
            console.error("Failed to set LLM config:", error);
            const msg = error instanceof Error ? error.message : "Model değiştirilirken hata oluştu!";
            alert(msg);
        } finally {
            setIsChangingModel(false);
        }
    };

    return (
        <header className="flex items-center justify-between h-14 px-4 sticky top-0 z-10 bg-background/80 backdrop-blur-md">
            <div className="flex items-center gap-2">
                {!isSidebarOpen && (
                    <Tooltip>
                        <TooltipTrigger asChild>
                            <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8 text-muted-foreground hover:text-foreground"
                                onClick={toggleSidebar}
                            >
                                <PanelLeft size={18} />
                            </Button>
                        </TooltipTrigger>
                        <TooltipContent>Kenar çubuğunu aç</TooltipContent>
                    </Tooltip>
                )}

                <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                        <Button
                            variant="ghost"
                            className="gap-2 text-sm font-semibold text-foreground/80 hover:text-foreground hover:bg-accent/60 rounded-xl h-9 px-3"
                            disabled={isChangingModel}
                        >
                            <span className="text-[15px]">{selectedModel}</span>
                            <ChevronDown size={14} className="text-muted-foreground" />
                        </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="start" className="w-56 rounded-xl border-border/60 shadow-lg">
                        <DropdownMenuItem
                            className="gap-3 cursor-pointer py-2 px-3"
                            onClick={() => handleModelChange("Frappe Pro", "vllm", "Qwen/Qwen2.5-1.5B-Instruct-AWQ")}
                        >
                            <Sparkles size={16} className={selectedModel === "Frappe Pro" ? "text-primary" : "text-foreground/40"} />
                            <div>
                                <p className="text-sm font-medium">Frappe Pro</p>
                                <p className="text-xs text-muted-foreground">vLLM Local</p>
                            </div>
                        </DropdownMenuItem>

                        {isMounted && (
                            <>
                                {settings.openaiKey && (
                                    <>
                                        <div className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground/60 border-t border-border/40 mt-1">OpenAI</div>
                                        <DropdownMenuItem
                                            className="gap-3 cursor-pointer py-2 px-3"
                                            onClick={() => handleModelChange("GPT-4o", "openai", "gpt-4o", settings.openaiKey)}
                                        >
                                            <Image src="/openai_icon.png" alt="OpenAI" width={16} height={16} className={`object-contain ${selectedModel !== "GPT-4o" && "opacity-60 grayscale"}`} />
                                            <div>
                                                <p className="text-sm font-medium">GPT-4o</p>
                                                <p className="text-xs text-muted-foreground">En yetenekli model</p>
                                            </div>
                                        </DropdownMenuItem>
                                        <DropdownMenuItem
                                            className="gap-3 cursor-pointer py-2 px-3"
                                            onClick={() => handleModelChange("GPT-4o Mini", "openai", "gpt-4o-mini", settings.openaiKey)}
                                        >
                                            <Image src="/openai_icon.png" alt="OpenAI Mini" width={16} height={16} className={`object-contain ${selectedModel !== "GPT-4o Mini" && "opacity-60 grayscale"}`} />
                                            <div>
                                                <p className="text-sm font-medium">GPT-4o Mini</p>
                                                <p className="text-xs text-muted-foreground">Hızlı ve ekonomik</p>
                                            </div>
                                        </DropdownMenuItem>
                                    </>
                                )}

                                {settings.anthropicKey && (
                                    <>
                                        <div className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground/60 border-t border-border/40 mt-1">Anthropic</div>
                                        <DropdownMenuItem
                                            className="gap-3 cursor-pointer py-2 px-3"
                                            onClick={() => handleModelChange("Claude 3.5 Sonnet", "litellm", "claude-3-5-sonnet-20240620", settings.anthropicKey)}
                                        >
                                            <Image src="/claude_icon.png" alt="Claude" width={16} height={16} className={`object-contain ${selectedModel !== "Claude 3.5 Sonnet" && "opacity-60 grayscale"}`} />
                                            <div>
                                                <p className="text-sm font-medium">Claude 3.5 Sonnet</p>
                                                <p className="text-xs text-muted-foreground">Akıllı ve dengeli</p>
                                            </div>
                                        </DropdownMenuItem>
                                    </>
                                )}

                                {settings.openRouterKey && (
                                    <>
                                        <div className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground/60 border-t border-border/40 mt-1">OpenRouter</div>
                                        <DropdownMenuItem
                                            className="gap-3 cursor-pointer py-2 px-3"
                                            onClick={() => handleModelChange("DeepSeek Coder", "litellm", "deepseek/deepseek-coder", settings.openRouterKey)}
                                        >
                                            <Image src="/openrouter-icon.webp" alt="OpenRouter" width={16} height={16} className={`object-contain ${selectedModel !== "DeepSeek Coder" && "opacity-60 grayscale"}`} />
                                            <div>
                                                <p className="text-sm font-medium">DeepSeek Coder</p>
                                                <p className="text-xs text-muted-foreground">Uzman yazılımcı</p>
                                            </div>
                                        </DropdownMenuItem>
                                    </>
                                )}

                                {settings.geminiKey && (
                                    <>
                                        <div className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground/60 border-t border-border/40 mt-1">Google</div>
                                        <DropdownMenuItem
                                            className="gap-3 cursor-pointer py-2 px-3"
                                            onClick={() => handleModelChange("Gemini 1.5 Pro", "litellm", "gemini/gemini-1.5-pro", settings.geminiKey)}
                                        >
                                            <Image src="/gemini_icon.webp" alt="Gemini" width={16} height={16} className={`object-contain ${selectedModel !== "Gemini 1.5 Pro" && "opacity-60 grayscale"}`} />
                                            <div>
                                                <p className="text-sm font-medium">Gemini 1.5 Pro</p>
                                                <p className="text-xs text-muted-foreground">Google altyapısı</p>
                                            </div>
                                        </DropdownMenuItem>
                                    </>
                                )}

                                {settings.ollamaUrl && (
                                    <>
                                        <div className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider text-muted-foreground/60 border-t border-border/40 mt-1">Ollama Local</div>
                                        <DropdownMenuItem
                                            className="gap-3 cursor-pointer py-2 px-3"
                                            onClick={() => handleModelChange("Llama 3 (8B)", "litellm", "ollama/llama3", undefined, settings.ollamaUrl)}
                                        >
                                            <Image src="/ollama-icon.webp" alt="Ollama" width={16} height={16} className={`object-contain ${selectedModel !== "Llama 3 (8B)" && "opacity-60 grayscale"}`} />
                                            <div>
                                                <p className="text-sm font-medium">Llama 3</p>
                                                <p className="text-xs text-muted-foreground">Yerel model</p>
                                            </div>
                                        </DropdownMenuItem>
                                    </>
                                )}
                            </>
                        )}
                    </DropdownMenuContent>
                </DropdownMenu>

            </div>

            <div className="flex items-center gap-1.5">
                <Tooltip>
                    <TooltipTrigger asChild>
                        <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 rounded-full text-foreground/60 hover:text-foreground hover:bg-accent"
                            onClick={handleToggleTheme}
                        >
                            {isDark ? <Sun size={18} /> : <Moon size={18} />}
                        </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                        {isDark ? "Açık Tema" : "Koyu Tema"}
                    </TooltipContent>
                </Tooltip>

                <SettingsDialog>
                    <div>
                        <Tooltip>
                            <TooltipTrigger asChild>
                                <Button variant="ghost" size="icon" className="h-8 w-8 rounded-full text-foreground/60 hover:text-foreground hover:bg-accent">
                                    <Settings size={18} />
                                </Button>
                            </TooltipTrigger>
                            <TooltipContent>Ayarlar</TooltipContent>
                        </Tooltip>
                    </div>
                </SettingsDialog>
            </div>
        </header>
    );
}
