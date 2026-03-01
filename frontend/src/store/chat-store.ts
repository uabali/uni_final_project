import { create } from "zustand";

export interface Message {
    id: string;
    role: "user" | "assistant" | "system";
    content: string;
    timestamp: Date;
    sources?: { title: string; url: string; snippet?: string }[];
}

export interface Conversation {
    id: string;
    title: string;
    messages: Message[];
    createdAt: Date;
    updatedAt: Date;
}

interface ChatState {
    conversations: Conversation[];
    activeConversationId: string | null;
    isGenerating: boolean;
    isSidebarOpen: boolean;

    // Actions
    createConversation: () => string;
    deleteConversation: (id: string) => void;
    setActiveConversation: (id: string) => void;
    addMessage: (conversationId: string, message: Omit<Message, "id" | "timestamp">) => void;
    addMessageWithId: (conversationId: string, message: Omit<Message, "timestamp">) => void;
    updateMessage: (conversationId: string, messageId: string, content: string) => void;
    appendMessageContent: (conversationId: string, messageId: string, contentDelta: string) => void;
    setMessageSources: (conversationId: string, messageId: string, sources: { title: string; url: string; snippet?: string }[]) => void;
    setIsGenerating: (value: boolean) => void;
    toggleSidebar: () => void;
    setSidebarOpen: (value: boolean) => void;
}

const generateId = () => Math.random().toString(36).substring(2, 15) + Date.now().toString(36);

export const useChatStore = create<ChatState>((set, get) => ({
    conversations: [],
    activeConversationId: null,
    isGenerating: false,
    isSidebarOpen: true,

    createConversation: () => {
        const id = generateId();
        const newConversation: Conversation = {
            id,
            title: "Yeni Sohbet",
            messages: [],
            createdAt: new Date(),
            updatedAt: new Date(),
        };
        set((state) => ({
            conversations: [newConversation, ...state.conversations],
            activeConversationId: id,
        }));
        return id;
    },

    deleteConversation: (id) => {
        set((state) => {
            const filtered = state.conversations.filter((c) => c.id !== id);
            return {
                conversations: filtered,
                activeConversationId:
                    state.activeConversationId === id
                        ? filtered.length > 0
                            ? filtered[0].id
                            : null
                        : state.activeConversationId,
            };
        });
    },

    setActiveConversation: (id) => {
        set({ activeConversationId: id });
    },

    addMessage: (conversationId, message) => {
        const newMessage: Message = {
            ...message,
            id: generateId(),
            timestamp: new Date(),
        };
        set((state) => ({
            conversations: state.conversations.map((conv) => {
                if (conv.id !== conversationId) return conv;
                const updatedMessages = [...conv.messages, newMessage];
                // Auto-title from first user message
                const title =
                    conv.messages.length === 0 && message.role === "user"
                        ? message.content.substring(0, 40) + (message.content.length > 40 ? "..." : "")
                        : conv.title;
                return {
                    ...conv,
                    messages: updatedMessages,
                    title,
                    updatedAt: new Date(),
                };
            }),
        }));
    },

    addMessageWithId: (conversationId, message) => {
        const newMessage: Message = {
            ...message,
            timestamp: new Date(),
        };
        set((state) => ({
            conversations: state.conversations.map((conv) => {
                if (conv.id !== conversationId) return conv;
                const updatedMessages = [...conv.messages, newMessage];
                const title =
                    conv.messages.length === 0 && message.role === "user"
                        ? message.content.substring(0, 40) + (message.content.length > 40 ? "..." : "")
                        : conv.title;
                return {
                    ...conv,
                    messages: updatedMessages,
                    title,
                    updatedAt: new Date(),
                };
            }),
        }));
    },

    updateMessage: (conversationId, messageId, content) => {
        set((state) => ({
            conversations: state.conversations.map((conv) => {
                if (conv.id !== conversationId) return conv;
                return {
                    ...conv,
                    messages: conv.messages.map((msg) =>
                        msg.id === messageId ? { ...msg, content } : msg
                    ),
                    updatedAt: new Date(),
                };
            }),
        }));
    },

    appendMessageContent: (conversationId, messageId, contentDelta) => {
        set((state) => ({
            conversations: state.conversations.map((conv) => {
                if (conv.id !== conversationId) return conv;
                return {
                    ...conv,
                    messages: conv.messages.map((msg) =>
                        msg.id === messageId ? { ...msg, content: msg.content + contentDelta } : msg
                    ),
                    updatedAt: new Date(),
                };
            }),
        }));
    },

    setIsGenerating: (value) => set({ isGenerating: value }),

    setMessageSources: (conversationId, messageId, sources) => {
        set((state) => ({
            conversations: state.conversations.map((conv) => {
                if (conv.id !== conversationId) return conv;
                return {
                    ...conv,
                    messages: conv.messages.map((msg) =>
                        msg.id === messageId ? { ...msg, sources } : msg
                    ),
                    updatedAt: new Date(),
                };
            }),
        }));
    },

    toggleSidebar: () => set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),

    setSidebarOpen: (value) => set({ isSidebarOpen: value }),
}));
