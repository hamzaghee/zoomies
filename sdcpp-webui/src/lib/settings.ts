import { type Ref, ref, watch } from "vue";

export function normalizePollIntervalMs(value: unknown): number {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
        return 100;
    }
    return Math.max(1, Math.round(numeric));
}

export type ThemeChoice = "system" | "light" | "dark";

export const THEME_CHOICES: ThemeChoice[] = ["system", "light", "dark"];

export function normalizeThemeChoice(value: unknown): ThemeChoice {
    return THEME_CHOICES.includes(value as ThemeChoice) ? (value as ThemeChoice) : "system";
}

export function createStoredRef<T>(key: string, fallbackValue: T, normalize: (value: any) => T = (value) => value): Ref<T> {
    const state = ref(readStoredValue(key, fallbackValue, normalize)) as Ref<T>;

    watch(state, (value) => {
        const normalized = normalize(value);
        if (normalized !== value) {
            state.value = normalized;
            return;
        }
        window.localStorage.setItem(key, JSON.stringify(normalized));
    });

    return state;
}

function readStoredValue<T>(key: string, fallbackValue: T, normalize: (value: any) => T): T {
    try {
        const storedValue = window.localStorage.getItem(key);
        if (storedValue == null) {
            return normalize(fallbackValue);
        }
        return normalize(JSON.parse(storedValue));
    } catch {
        return normalize(fallbackValue);
    }
}
