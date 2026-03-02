import { create } from "zustand";

export interface TaskStatus {
    task_id: string;
    status: string;
    progress?: number | null;
    last_error?: string | null;
    updated_at: string;
    [key: string]: any;
}

interface TaskState {
    tasks: Record<string, TaskStatus>;
    upsertTask: (task: TaskStatus) => void;
    clearTask: (taskId: string) => void;
}

export const useTaskStore = create<TaskState>()((set) => ({
    tasks: {},
    upsertTask: (task) =>
        set((state) => ({
            tasks: {
                ...state.tasks,
                [task.task_id]: task,
            },
        })),
    clearTask: (taskId) =>
        set((state) => {
            const next = { ...state.tasks };
            delete next[taskId];
            return { tasks: next };
        }),
}));

