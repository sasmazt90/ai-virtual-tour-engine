import { X } from "lucide-react";

export function ModalShell({ title, children, onClose }) {
  return (
    <div className="fixed inset-0 z-[100]">
      <div className="absolute inset-0 bg-black/60" onClick={onClose}></div>
      <div className="absolute inset-0 flex items-center justify-center p-4">
        {/**
         * IMPORTANT:
         * In light theme, we should NOT rely on `dark:*` classes here.
         * Tailwind dark utilities can still apply when OS theme is dark even if our app theme is light.
         * Use ThemeProvider CSS variables instead.
         */}
        <div className="w-full max-w-2xl rounded-2xl border border-[var(--border-color)] bg-[var(--card-bg)] shadow-2xl overflow-hidden backdrop-blur">
          <div className="flex items-center justify-between px-5 py-4 border-b border-[var(--border-color)]">
            <div className="text-base sm:text-lg font-semibold font-jetbrains-mono text-[var(--text-primary)]">
              {title}
            </div>
            <button
              type="button"
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-black/5 transition-colors text-[var(--text-secondary)]"
              aria-label="Close"
            >
              <X size={18} />
            </button>
          </div>

          <div className="p-5">{children}</div>
        </div>
      </div>
    </div>
  );
}
