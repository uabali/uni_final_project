"use client";

import React, { useRef, useState, useMemo } from "react";
import { useChatStore, Conversation } from "@/store/chat-store";
import { useDataStore } from "@/store/data-store";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
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
import {
    Plus,
    MessageSquare,
    Trash2,
    MoreHorizontal,
    PanelLeftClose,
    Settings,
    Database,
    UploadCloud,
} from "lucide-react";
import Image from "next/image";
import { SettingsDialog } from "./settings-dialog";
import { uploadFiles, deleteIngestedFiles } from "@/lib/api";
import { useTaskStore } from "@/store/task-store";
import { Cpu } from "lucide-react";

function formatRelativeDate(date: Date): string {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return "Şu an";
    if (minutes < 60) return `${minutes}dk önce`;
    if (hours < 24) return `${hours}sa önce`;
    if (days < 7) return `${days}g önce`;
    return date.toLocaleDateString("tr-TR", { month: "short", day: "numeric" });
}

function ConversationItem({
    conversation,
    isActive,
}: {
    conversation: Conversation;
    isActive: boolean;
}) {
    const { setActiveConversation, deleteConversation } = useChatStore();

    return (
        <div
            role="button"
            tabIndex={0}
            className={`
        group relative flex items-center gap-3 rounded-[10px] px-3 py-2 cursor-pointer
        transition-all duration-200 ease-out mx-2
        ${isActive
                    ? "bg-accent/80 text-foreground"
                    : "hover:bg-accent/50 text-muted-foreground hover:text-foreground"
                }
      `}
            onClick={() => setActiveConversation(conversation.id)}
            onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                    setActiveConversation(conversation.id);
                }
            }}
        >
            <div className="flex-1 min-w-0">
                <p className="text-[14px] truncate tracking-tight pr-6">
                    {conversation.title}
                </p>
            </div>

            <DropdownMenu>
                <DropdownMenuTrigger asChild>
                    <button
                        className={`
              absolute right-2 px-1 rounded-md transition-all duration-200
              text-muted-foreground/70 hover:text-foreground hover:bg-background/60
            `}
                        onClick={(e) => e.stopPropagation()}
                    >
                        <MoreHorizontal size={15} />
                    </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-40 rounded-xl">
                    <DropdownMenuItem
                        className="text-destructive focus:text-destructive cursor-pointer rounded-lg px-3 py-2"
                        onClick={(e) => {
                            e.stopPropagation();
                            deleteConversation(conversation.id);
                        }}
                    >
                        <Trash2 size={15} className="mr-2" />
                        Sohbeti Sil
                    </DropdownMenuItem>
                </DropdownMenuContent>
            </DropdownMenu>
        </div>
    );
}

export function Sidebar() {
    const {
        conversations,
        activeConversationId,
        createConversation,
        isSidebarOpen,
        toggleSidebar,
    } = useChatStore();

    const isCollapsed = !isSidebarOpen;
    const { files: dataFiles, addFiles, removeFile } = useDataStore();
    const [isUploadingData, setIsUploadingData] = useState(false);
    const dataInputRef = useRef<HTMLInputElement>(null);

    const handleDataSelect = async (
        event: React.ChangeEvent<HTMLInputElement>
    ) => {
        const fileList = event.target.files;
        if (!fileList || fileList.length === 0) return;
        const files = Array.from(fileList);

        setIsUploadingData(true);
        try {
            await uploadFiles(files);
            addFiles(files.map((f) => ({ name: f.name, size: f.size })));
        } catch (error) {
            console.error("Data upload failed:", error);
        } finally {
            setIsUploadingData(false);
            // same file yeniden seçilebilsin
            if (dataInputRef.current) {
                dataInputRef.current.value = "";
            }
        }
    };

    const handleDataDelete = async (name: string) => {
        try {
            await deleteIngestedFiles([name]);
            removeFile(name);
        } catch (error) {
            console.error("Data delete failed:", error);
        }
    };

    // Group conversations by time
    const today: Conversation[] = [];
    const yesterday: Conversation[] = [];
    const thisWeek: Conversation[] = [];
    const older: Conversation[] = [];

    const now = new Date();
    conversations.forEach((conv) => {
        const diff = now.getTime() - conv.updatedAt.getTime();
        const days = Math.floor(diff / 86400000);
        if (days === 0) today.push(conv);
        else if (days === 1) yesterday.push(conv);
        else if (days < 7) thisWeek.push(conv);
        else older.push(conv);
    });

    const groups = [
        { label: "Bugün", items: today },
        { label: "Dün", items: yesterday },
        { label: "Son 7 Gün", items: thisWeek },
        { label: "Önceki 30 Gün", items: older },
    ].filter((g) => g.items.length > 0);

    // Agent status panel icin aktif conversation'a ait task'i sec
    const { tasks } = useTaskStore();
    const activeTask = useMemo(() => {
        if (!activeConversationId) return null;
        return tasks[activeConversationId] || null;
    }, [tasks, activeConversationId]);

    return (
        <aside
            className={`
        flex flex-col h-full bg-[#f9f9f9] dark:bg-sidebar border-r border-border/40
        transition-all duration-300 ease-in-out
        ${isCollapsed ? "w-[72px]" : "w-[260px]"}
      `}
        >
            {/* Header */}
            <div
                className={`
                flex items-center p-3.5 pb-2
                ${isCollapsed ? "justify-center" : "justify-between"}
            `}
            >
                <div
                    className={`
                    flex items-center
                    ${isCollapsed ? "" : "gap-2.5 pl-1.5"}
                `}
                >
                    <Image src="/frappe_icon.png" alt="Logo" width={26} height={26} className="rounded-md object-contain" />
                    {!isCollapsed && (
                        <h1 className="ml-2 text-[17px] font-semibold tracking-tight text-foreground/90">
                            Frappe
                        </h1>
                    )}
                </div>
                {!isCollapsed && (
                    <Tooltip>
                        <TooltipTrigger asChild>
                            <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8 rounded-md text-muted-foreground hover:text-foreground hover:bg-black/5 dark:hover:bg-white/5"
                                onClick={toggleSidebar}
                            >
                                <PanelLeftClose size={18} strokeWidth={1.75} />
                            </Button>
                        </TooltipTrigger>
                        <TooltipContent side="right">Kenar çubuğunu kapat</TooltipContent>
                    </Tooltip>
                )}
            </div>

            {/* New Chat Button & Conversation List - hidden when collapsed */}
            {!isCollapsed && (
                <>
                    {/* Hidden input for Data uploads */}
                    <input
                        ref={dataInputRef}
                        type="file"
                        multiple
                        className="hidden"
                        onChange={handleDataSelect}
                        accept=".pdf,.txt"
                    />

                    <div className="px-3 py-2.5">
                        <Button
                            variant="outline"
                            className="w-full justify-between items-center rounded-xl bg-background hover:bg-accent/50 border border-border/60 hover:border-border transition-all duration-200 h-9 font-medium text-foreground/80"
                            onClick={createConversation}
                        >
                            <span className="text-[14px]">Yeni Sohbet</span>
                            <Plus size={16} strokeWidth={2} />
                        </Button>
                    </div>

                    {/* Data Area */}
                    <div className="px-3 pb-2">
                        <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2 text-foreground/80">
                                <Database size={15} />
                                <span className="text-[13px] font-semibold tracking-tight">
                                    Data Alanı
                                </span>
                            </div>
                            <Button
                                variant="ghost"
                                size="icon"
                                className="h-7 w-7 rounded-full text-muted-foreground hover:text-foreground hover:bg-black/5 dark:hover:bg-white/5"
                                disabled={isUploadingData}
                                onClick={() => dataInputRef.current?.click()}
                            >
                                <UploadCloud size={15} />
                            </Button>
                        </div>
                        <div className="rounded-lg bg-background/70 dark:bg-black/20 border border-border/50 max-h-32 overflow-auto">
                            {dataFiles.length === 0 ? (
                                <p className="px-3 py-2 text-[11px] text-muted-foreground/70">
                                    PDF ve TXT dosyalarını buradan yükleyin. RAG modu açıkken
                                    bu dokümanlar üzerinden yanıt üretilir.
                                </p>
                            ) : (
                                <ul className="py-1.5">
                                    {dataFiles.map((file) => (
                                        <li
                                            key={file.name}
                                            className="px-3 py-1 text-[11px] text-foreground/80 flex items-center justify-between gap-2 group"
                                        >
                                            <span className="truncate" title={file.name}>
                                                {file.name}
                                            </span>
                                            <div className="flex items-center gap-1.5">
                                                <span className="text-[10px] text-muted-foreground/70">
                                                    {Math.round(file.size / 1024)} KB
                                                </span>
                                                <button
                                                    type="button"
                                                    onClick={() => handleDataDelete(file.name)}
                                                    className="p-1 rounded-full text-muted-foreground/60 hover:text-destructive hover:bg-destructive/10 opacity-0 group-hover:opacity-100 transition-opacity"
                                                >
                                                    <Trash2 size={11} />
                                                </button>
                                            </div>
                                        </li>
                                    ))}
                                </ul>
                            )}
                        </div>
                    </div>

                    <div className="px-4">
                        <Separator className="my-1 opacity-50 bg-border/40" />
                    </div>

                    {/* Conversation List */}
                    <ScrollArea className="flex-1">
                        <div className="py-2.5 space-y-5">
                            {groups.length === 0 && (
                                <div className="flex flex-col items-center justify-center py-12 px-4 text-center mt-6">
                                    <div className="w-10 h-10 rounded-full bg-accent/80 flex items-center justify-center mb-3 text-muted-foreground/60">
                                        <MessageSquare size={18} />
                                    </div>
                                    <p className="text-[13px] text-muted-foreground font-medium">Sohbet geçmişi yok</p>
                                </div>
                            )}

                            {groups.map((group) => (
                                <div key={group.label}>
                                    <p className="px-5 py-2 text-[11px] font-semibold text-muted-foreground/50 uppercase tracking-widest">
                                        {group.label}
                                    </p>
                                    <div className="space-y-0.5">
                                        {group.items.map((conv) => (
                                            <ConversationItem
                                                key={conv.id}
                                                conversation={conv}
                                                isActive={conv.id === activeConversationId}
                                            />
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </ScrollArea>
                </>
            )}

            {/* Footer - Settings & User Profile */}
            <div className="p-3 space-y-1">
                {/* Agent Status (yalnızca geniş görünümde) */}
                {!isCollapsed && (
                    <div className="mb-1">
                        <div className="flex items-center gap-2 px-3 pb-1">
                            <Cpu size={14} className="text-muted-foreground" />
                            <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground/70">
                                Agent Durumu
                            </span>
                        </div>
                        <div className="mx-3 rounded-xl border border-border/60 bg-background/80 px-3 py-2 text-[11px] text-muted-foreground/80">
                            {activeTask ? (
                                <div className="space-y-1">
                                    <div className="flex items-center justify-between">
                                        <span className="font-semibold text-foreground/80">
                                            {activeTask.status.toUpperCase()}
                                        </span>
                                        {typeof activeTask.progress === "number" && (
                                            <span>
                                                %{Math.round(activeTask.progress * 100)}
                                            </span>
                                        )}
                                    </div>
                                    <div className="text-[10px] text-muted-foreground/70">
                                        Son güncelleme:{" "}
                                        {new Date(activeTask.updated_at).toLocaleTimeString("tr-TR", {
                                            hour: "2-digit",
                                            minute: "2-digit",
                                        })}
                                    </div>
                                    {activeTask.last_error && (
                                        <div className="text-[10px] text-red-500/80 truncate">
                                            Hata: {activeTask.last_error}
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <span className="text-[10px] text-muted-foreground/70">
                                    Aktif bir uzun süreli görev yok.
                                </span>
                            )}
                        </div>
                    </div>
                )}

                {/* Settings Button */}
                <SettingsDialog>
                    <div
                        className={`
                        flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-black/5 dark:hover:bg-white/5 transition-colors cursor-pointer text-foreground/80
                        ${isCollapsed ? "justify-center" : ""}
                    `}
                    >
                        <Settings size={18} className="text-muted-foreground" />
                        {!isCollapsed && (
                            <p className="text-[14px] font-medium truncate tracking-tight">Ayarlar</p>
                        )}
                    </div>
                </SettingsDialog>

                {/* User Profile */}
                <div
                    className={`
                    flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-black/5 dark:hover:bg-white/5 transition-colors cursor-pointer text-foreground/80
                    ${isCollapsed ? "justify-center" : ""}
                `}
                >
                    <div className="w-7 h-7 rounded-sm bg-gradient-to-br from-primary/80 to-primary text-primary-foreground flex items-center justify-center font-bold text-xs drop-shadow-sm">
                        U
                    </div>
                    {!isCollapsed && (
                        <div className="flex-1 min-w-0">
                            <p className="text-[14px] font-semibold truncate tracking-tight">Üye Kullanıcı</p>
                        </div>
                    )}
                </div>
            </div>
        </aside>
    );
}
