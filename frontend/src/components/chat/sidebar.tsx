"use client";

import React from "react";
import { useChatStore, Conversation } from "@/store/chat-store";
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
} from "lucide-react";
import Image from "next/image";

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
              ${isActive ? "opacity-100" : "opacity-0 group-hover:opacity-100"}
              hover:bg-background/60 text-muted-foreground hover:text-foreground
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

    return (
        <aside
            className={`
        flex flex-col h-full bg-[#f9f9f9] dark:bg-sidebar border-r border-border/40
        transition-all duration-300 ease-in-out
        ${isSidebarOpen ? "w-[260px]" : "w-0 overflow-hidden border-r-0"}
      `}
        >
            {/* Header */}
            <div className="flex items-center justify-between p-3.5 pb-2">
                <div className="flex items-center gap-2.5 pl-1.5">
                    <Image src="/frappe_icon.png" alt="Logo" width={26} height={26} className="rounded-md object-contain" />
                    <h1 className="text-[17px] font-semibold tracking-tight text-foreground/90">
                        Frappe
                    </h1>
                </div>
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
            </div>

            {/* New Chat Button */}
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

            {/* Footer User Profile */}
            <div className="p-3">
                <div className="flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-black/5 dark:hover:bg-white/5 transition-colors cursor-pointer text-foreground/80">
                    <div className="w-7 h-7 rounded-sm bg-gradient-to-br from-primary/80 to-primary text-primary-foreground flex items-center justify-center font-bold text-xs drop-shadow-sm">
                        U
                    </div>
                    <div className="flex-1 min-w-0">
                        <p className="text-[14px] font-semibold truncate tracking-tight">Üye Kullanıcı</p>
                    </div>
                </div>
            </div>
        </aside>
    );
}
