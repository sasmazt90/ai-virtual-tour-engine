import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

const ThemeContext = createContext({
  theme: "dark",
  setTheme: () => {},
  toggleTheme: () => {},
});

const STORAGE_KEY = "estate_theme";

export function ThemeProvider({ children, brandStyleVars }) {
  const [theme, setTheme] = useState("dark");

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    try {
      const saved = window.localStorage.getItem(STORAGE_KEY);
      if (saved === "light" || saved === "dark") {
        setTheme(saved);
        return;
      }

      // Default: dark, unless user explicitly prefers light.
      const prefersLight =
        window.matchMedia &&
        window.matchMedia("(prefers-color-scheme: light)").matches;

      setTheme(prefersLight ? "light" : "dark");
    } catch {
      // default remains dark
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    try {
      window.localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      // ignore
    }
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((t) => (t === "dark" ? "light" : "dark"));
  }, []);

  const value = useMemo(() => {
    return { theme, setTheme, toggleTheme };
  }, [theme, toggleTheme]);

  const background =
    theme === "dark"
      ? "radial-gradient(900px circle at 15% 25%, rgba(99,102,241,0.16), transparent 55%), radial-gradient(900px circle at 85% 15%, rgba(16,185,129,0.10), transparent 55%), linear-gradient(180deg, #07080A 0%, #050608 100%)"
      : "radial-gradient(900px circle at 15% 25%, rgba(99,102,241,0.12), transparent 55%), radial-gradient(900px circle at 85% 15%, rgba(16,185,129,0.09), transparent 55%), linear-gradient(180deg, #ffffff 0%, #f6f7fb 100%)";

  // Dedicated LIGHT MODE tokens (critical for readability)
  const lightTokens = {
    "--text-primary-light": "#111111",
    "--text-secondary-light": "#5A5A5A",
    "--card-bg-light": "#FFFFFF",
    "--border-light": "#E5E5E5",
  };

  const darkTokens = {
    "--text-primary-dark": "rgba(255,255,255,0.92)",
    "--text-secondary-dark": "rgba(255,255,255,0.72)",
    "--card-bg-dark": "rgba(255,255,255,0.06)",
    "--border-dark": "rgba(255,255,255,0.10)",
  };

  const activeTokens =
    theme === "light"
      ? {
          "--text-primary": "var(--text-primary-light)",
          "--text-secondary": "var(--text-secondary-light)",
          "--card-bg": "var(--card-bg-light)",
          "--border-color": "var(--border-light)",
          // UI helpers for inputs/hover states
          "--surface-muted": "rgba(0,0,0,0.06)",
          "--surface-hover": "rgba(0,0,0,0.05)",
        }
      : {
          "--text-primary": "var(--text-primary-dark)",
          "--text-secondary": "var(--text-secondary-dark)",
          "--card-bg": "var(--card-bg-dark)",
          "--border-color": "var(--border-dark)",
          // UI helpers for inputs/hover states
          "--surface-muted": "rgba(255,255,255,0.08)",
          "--surface-hover": "rgba(255,255,255,0.10)",
        };

  const themeClassName = theme === "dark" ? "dark theme-dark" : "theme-light";

  const lightGrainStyle = {
    // Light mode MUST prioritize readability.
    opacity: 0.025,
    mixBlendMode: "normal",
    backgroundImage:
      "radial-gradient(circle at 25% 20%, rgba(0, 0, 0, 0.03) 0, rgba(0, 0, 0, 0) 55%), radial-gradient(circle at 75% 60%, rgba(0, 0, 0, 0.02) 0, rgba(0, 0, 0, 0) 60%)",
  };

  const lightModeTailwindOverrideCss = useMemo(() => {
    if (theme !== "light") {
      return "";
    }

    // IMPORTANT:
    // If Tailwind is configured with darkMode: 'media', `dark:*` utility classes
    // still apply when the OS theme is dark, even if our app theme is 'light'.
    // These overrides force readable colors inside `.theme-light`.
    return `
      /* Dedicated tokens (requested) */
      .theme-light { color: var(--text-primary-light); }

      /* Text: never allow dark-mode colors to leak into light mode */
      .theme-light .dark\\:text-gray-50 { color: var(--text-primary-light) !important; }
      .theme-light .dark\\:text-gray-100 { color: var(--text-primary-light) !important; }
      .theme-light .dark\\:text-gray-200 { color: var(--text-secondary-light) !important; }
      .theme-light .dark\\:text-gray-300 { color: var(--text-secondary-light) !important; }
      .theme-light .dark\\:text-gray-400 { color: var(--text-secondary-light) !important; }
      .theme-light .dark\\:text-gray-500 { color: var(--text-secondary-light) !important; }
      .theme-light .dark\\:text-white { color: var(--text-primary-light) !important; }

      /* Common text utilities used across the app */
      .theme-light .text-gray-950 { color: var(--text-primary-light) !important; }
      .theme-light .text-gray-900 { color: var(--text-primary-light) !important; }
      .theme-light .text-gray-800 { color: var(--text-primary-light) !important; }
      .theme-light .text-gray-700 { color: var(--text-secondary-light) !important; }
      .theme-light .text-gray-600 { color: var(--text-secondary-light) !important; }
      .theme-light .text-gray-500 { color: var(--text-secondary-light) !important; }
      .theme-light .text-gray-400 { color: var(--text-secondary-light) !important; }

      .theme-light .placeholder-gray-500::placeholder { color: rgba(90, 90, 90, 0.70) !important; }
      .theme-light .dark\\:placeholder-gray-400::placeholder { color: rgba(90, 90, 90, 0.70) !important; }

      /* Borders */
      .theme-light .dark\\:border-white\\/10 { border-color: var(--border-light) !important; }
      .theme-light .dark\\:border-white\\/12 { border-color: var(--border-light) !important; }
      .theme-light .border-black\\/10 { border-color: var(--border-light) !important; }

      /* Backgrounds (cards / panels) */
      .theme-light .dark\\:bg-\\[\\#07080A\\] { background-color: var(--card-bg-light) !important; }
      .theme-light .dark\\:bg-\\[\\#050608\\] { background-color: var(--card-bg-light) !important; }
      .theme-light .dark\\:bg-\\[\\#1E1E1E\\] { background-color: #F6F7FB !important; }
      .theme-light .dark\\:bg-\\[\\#262626\\] { background-color: var(--card-bg-light) !important; }
      .theme-light .dark\\:bg-gray-800 { background-color: #F6F7FB !important; }
      .theme-light .dark\\:bg-gray-700 { background-color: #FFFFFF !important; }

      .theme-light .dark\\:bg-white\\/5 { background-color: rgba(0,0,0,0.05) !important; }
      .theme-light .dark\\:bg-white\\/10 { background-color: rgba(0,0,0,0.08) !important; }
      .theme-light .dark\\:bg-black\\/\\[0\\.45\\] { background-color: rgba(255,255,255,0.82) !important; }

      .theme-light .dark\\:shadow-\\[0_18px_60px_rgba\\(0\\,0\\,0\\,0\\.35\\)\\] { 
        box-shadow: 0 18px 60px rgba(0,0,0,0.16) !important;
      }

      /* Readability guardrails for app pages */
      .theme-light .ui-surface { color: var(--text-primary-light); }
      .theme-light .ui-surface .bg-white\\/70 { background-color: rgba(255,255,255,0.96) !important; }
      .theme-light .ui-surface .bg-white\\/80 { background-color: rgba(255,255,255,0.98) !important; }
      .theme-light .ui-surface .bg-white\\/95 { background-color: rgba(255,255,255,0.99) !important; }
    `;
  }, [theme]);

  // Global styles for native select <option> elements (browser-dependent rendering)
  const selectOptionCss = useMemo(() => {
    if (theme === "dark") {
      return `
        .dark select,
        .theme-dark select {
          color-scheme: dark;
        }
        .dark select option,
        .theme-dark select option {
          background-color: #1f2937 !important;
          color: #f3f4f6 !important;
        }
        .dark select optgroup,
        .theme-dark select optgroup {
          background-color: #1f2937 !important;
          color: #f3f4f6 !important;
        }
      `;
    }
    return `
      .theme-light select {
        color-scheme: light;
      }
      .theme-light select option {
        background-color: #ffffff !important;
        color: #111827 !important;
      }
      .theme-light select optgroup {
        background-color: #ffffff !important;
        color: #111827 !important;
      }
    `;
  }, [theme]);

  return (
    <ThemeContext.Provider value={value}>
      <div
        className={themeClassName}
        style={{
          ...(brandStyleVars || {}),
          "--app-bg": background,
          ...lightTokens,
          ...darkTokens,
          ...activeTokens,
          colorScheme: theme,
        }}
      >
        {lightModeTailwindOverrideCss ? (
          <style>{lightModeTailwindOverrideCss}</style>
        ) : null}
        <style>{selectOptionCss}</style>

        <div
          suppressHydrationWarning
          className={"relative min-h-screen text-gray-900 dark:text-gray-100"}
          style={{ background: "var(--app-bg)", color: "var(--text-primary)" }}
        >
          {/* Global subtle texture: strong in dark, extremely subtle in light (and never hurts readability). */}
          {theme === "dark" ? (
            <div className="absolute inset-0 pointer-events-none hero-grain" />
          ) : (
            <div
              className="absolute inset-0 pointer-events-none"
              style={lightGrainStyle}
            />
          )}

          {children}
        </div>
      </div>
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}
