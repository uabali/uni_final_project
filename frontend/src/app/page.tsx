"use client";

import React, { useState, useCallback, useEffect } from "react";
import { Sidebar } from "@/components/chat/sidebar";
import { ChatArea } from "@/components/chat/chat-area";
import { ChatHeader } from "@/components/chat/chat-header";
import { MessageInput } from "@/components/chat/message-input";
import { useChatStore } from "@/store/chat-store";
import { useSettingsStore } from "@/store/settings-store";
import Image from "next/image";
import { chatStream, setBackendApiKey } from "@/lib/api";

export default function Home() {
  const [suggestionText, setSuggestionText] = useState<string | undefined>();
  const { activeConversationId, conversations, createConversation, addMessage, addMessageWithId, appendMessageContent, setMessageSources, setIsGenerating } = useChatStore();
  const backendApiKey = useSettingsStore((s) => s.backendApiKey);

  useEffect(() => {
    setBackendApiKey(backendApiKey);
  }, [backendApiKey]);

  const activeConversation = conversations.find(c => c.id === activeConversationId);
  const isWelcome = !activeConversation || activeConversation.messages.length === 0;

  const handleSuggestionClick = useCallback(
    async (text: string) => {
      let convId = activeConversationId;
      if (!convId) {
        convId = createConversation();
      }

      addMessage(convId, {
        role: "user",
        content: text,
      });

      setIsGenerating(true);
      const assistantMessageId = Math.random().toString(36).substring(2, 15) + Date.now().toString(36);

      addMessageWithId(convId, {
        id: assistantMessageId,
        role: "assistant",
        content: "",
      });

      try {
        await chatStream(text, convId, (event) => {
          if (event.type === "token") {
            appendMessageContent(convId, assistantMessageId, event.data.text);
          } else if (event.type === "sources") {
            const sources = Array.isArray(event.data)
              ? event.data.map((s: any) => ({
                title: s.title || s.source || "",
                url: s.url || s.source || "",
                snippet: s.snippet || "",
              }))
              : [];
            if (sources.length > 0) {
              setMessageSources(convId, assistantMessageId, sources);
            }
          } else if (event.type === "error") {
            appendMessageContent(convId, assistantMessageId, `\n\n**Hata:** ${event.data.error}`);
          }
        });
      } catch (error) {
        console.error("Stream failed:", error);
        appendMessageContent(convId, assistantMessageId, "\n\n**Bağlantı Hatası:** API'ye erişilemiyor.");
      } finally {
        setIsGenerating(false);
      }
    },
    [activeConversationId, createConversation, addMessage, addMessageWithId, appendMessageContent, setMessageSources, setIsGenerating]
  );

  return (
    <main className="flex h-screen w-screen overflow-hidden bg-background">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0 h-full relative">
        <ChatHeader />

        {isWelcome ? (
          <div className="flex-1 flex flex-col items-center justify-center px-4 overflow-y-auto">
            <div className="w-full max-w-3xl flex flex-col items-center mt-[-5vh]">
              <div className="mb-6 flex items-center justify-center">
                <Image src="/frappe_icon.png" alt="Frappe Logo" width={56} height={56} className="rounded-xl drop-shadow-sm" />
              </div>
              <h2 className="text-3xl md:text-4xl font-medium mb-8 text-foreground tracking-tight text-center">
                Nasıl yardımcı olabilirim?
              </h2>

              <div className="w-full">
                <MessageInput
                  externalValue={suggestionText}
                  onExternalValueConsumed={() => setSuggestionText(undefined)}
                />
              </div>

              <div className="flex flex-wrap justify-center gap-2 mt-6 max-w-2xl">
                {[
                  { icon: "📝", label: "Yazı Yaz", prompt: "Yapay zeka hakkında kısa bir blog yazısı hazırla" },
                  { icon: "💡", label: "Fikir Üret", prompt: "Hafta sonu yapılabilecek yazılım projeleri öner" },
                  { icon: "💻", label: "Kod Yaz", prompt: "React ile basit bir sayaç bileşeni oluştur" },
                  { icon: "📊", label: "Analiz Et", prompt: "RAG mimarisinin temel avantajlarını açıkla" }
                ].map((s) => (
                  <button
                    key={s.label}
                    onClick={() => handleSuggestionClick(s.prompt)}
                    className="flex items-center gap-2 px-4 py-2.5 rounded-full border border-border/80 bg-card/40 hover:bg-accent text-sm font-medium text-foreground/80 hover:text-foreground transition-colors"
                  >
                    <span>{s.icon}</span>
                    <span>{s.label}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <>
            <div className="flex-1 overflow-y-auto h-full px-2">
              <ChatArea />
            </div>
            <div className="w-full bg-gradient-to-t from-background via-background to-transparent pt-4">
              <MessageInput
                externalValue={suggestionText}
                onExternalValueConsumed={() => setSuggestionText(undefined)}
              />
            </div>
          </>
        )}
      </div>
    </main>
  );
}
