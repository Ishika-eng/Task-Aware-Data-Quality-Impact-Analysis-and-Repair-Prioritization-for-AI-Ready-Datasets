import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

/* ---------------------------------------------------------------- formatting */
export const dash = "–";
const isNil = (v: unknown): v is null | undefined => v === null || v === undefined || (typeof v === "number" && !Number.isFinite(v));

export const pct = (v: number | null | undefined, d = 1) => (isNil(v) ? dash : `${(v * 100).toFixed(d)}%`);
export const num = (v: number | null | undefined, d = 4) => (isNil(v) ? dash : v.toFixed(d));
export const signed = (v: number | null | undefined, d = 4) => (isNil(v) ? dash : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`);
export const int = (v: number | null | undefined) => (isNil(v) ? dash : Math.round(v).toLocaleString());
export const bytes = (n: number) => (n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1048576).toFixed(1)} MB`);
export const seconds = (s: number) => (s < 90 ? `${s.toFixed(0)}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`);
export const cellText = (v: unknown) => (isNil(v) ? dash : String(v));

/* ---------------------------------------------------------------- issue vocabulary (one colour per error type) */
export const ISSUE_COLOR: Record<string, string> = {
  label_errors: "var(--c-label)",
  label_noise: "var(--c-label)",
  feature_corruption: "var(--c-feature)",
  outliers: "var(--c-outliers)",
  missing_values: "var(--c-missing)",
  duplicates: "var(--c-duplicates)",
  inconsistency: "var(--c-inconsistency)",
  joint_outliers: "var(--c-joint)",
};
export const ISSUE_NAME: Record<string, string> = {
  label_errors: "Label errors",
  label_noise: "Label errors",
  feature_corruption: "Feature anomalies",
  outliers: "Outliers",
  missing_values: "Missing values",
  duplicates: "Duplicates",
  inconsistency: "Inconsistencies",
  joint_outliers: "Unusual combinations",
};
export const ISSUE_ORDER = ["label_errors", "feature_corruption", "outliers", "missing_values", "duplicates"];

export type Tone = "neutral" | "good" | "warn" | "bad" | "accent";

/** Colour meaning for the many status words the API produces. */
export function tone(s: string | null | undefined): Tone {
  switch ((s ?? "").toLowerCase()) {
    case "positive":
    case "good":
    case "auto-repair":
    case "applied":
    case "complete":
      return "good";
    case "weak":
    case "attention":
    case "flag for human review":
    case "flagged_for_review":
    case "partial":
      return "warn";
    case "negative":
    case "inverted":
    case "high uncertainty":
      return "bad";
    case "flagged_only":
      return "accent";
    default:
      return "neutral";
  }
}

export const prettyStatus = (s: string | null | undefined) =>
  (s ?? "").replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

/* ---------------------------------------------------------------- theme */
type Theme = "light" | "dark";
const ThemeCtx = createContext<{ theme: Theme; toggle: () => void }>({ theme: "light", toggle: () => {} });

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(() => (document.documentElement.classList.contains("dark") ? "dark" : "light"));
  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    try {
      localStorage.setItem("taskclean-theme", theme);
    } catch {
      /* storage can be unavailable (private window); the theme still works for this visit */
    }
  }, [theme]);
  return <ThemeCtx.Provider value={{ theme, toggle: () => setTheme((t) => (t === "dark" ? "light" : "dark")) }}>{children}</ThemeCtx.Provider>;
}
export const useTheme = () => useContext(ThemeCtx);

/* ---------------------------------------------------------------- last audit (so the sidebar can link back to it) */
const KEY = "taskclean-last-audit";
export const rememberAudit = (id: string) => {
  try {
    localStorage.setItem(KEY, id);
  } catch {
    /* ignore */
  }
};
export const lastAudit = () => {
  try {
    return localStorage.getItem(KEY);
  } catch {
    return null;
  }
};
