import { useState, useMemo } from "react";
import { Command } from "cmdk";

export function ClientCombobox({ value, onChange, clients, placeholder }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  const selected = useMemo(() => {
    const arr = Array.isArray(clients) ? clients : [];
    return arr.find((c) => String(c.id) === String(value)) || null;
  }, [clients, value]);

  const filtered = useMemo(() => {
    const q = String(query || "")
      .toLowerCase()
      .trim();
    const arr = Array.isArray(clients) ? clients : [];
    if (!q) return arr;

    return arr.filter((c) => {
      const name = String(c?.full_name || "").toLowerCase();
      const email = String(c?.email || "").toLowerCase();
      const phone = String(c?.phone || "").toLowerCase();
      return name.includes(q) || email.includes(q) || phone.includes(q);
    });
  }, [clients, query]);

  const selectedLabel = useMemo(() => {
    if (!selected) return "";
    const emailPart = selected.email ? ` • ${selected.email}` : "";
    const phonePart = selected.phone ? ` • ${selected.phone}` : "";
    return `${selected.full_name}${emailPart}${phonePart}`;
  }, [selected]);

  // IMPORTANT: Avoid `dark:*` utilities in interactive pieces like dropdowns.
  // Use ThemeProvider variables so dark-mode + light-mode stay readable.
  const buttonClass =
    "w-full rounded-lg border border-[var(--border-color)] bg-[var(--card-bg)] hover:bg-[var(--surface-hover)] transition-colors px-3 py-2 text-left font-jetbrains-mono text-sm text-[var(--text-primary)]";

  const placeholderClass = "text-[var(--text-secondary)]";

  const dropdownClass =
    "absolute z-20 mt-2 w-full rounded-xl border border-[var(--border-color)] bg-[var(--card-bg)] shadow-xl overflow-hidden";

  const inputClass =
    "w-full rounded-lg border border-[var(--border-color)] bg-[var(--surface-muted)] px-3 py-2 text-sm font-jetbrains-mono text-[var(--text-primary)] placeholder-[var(--text-secondary)] outline-none focus:ring-2 focus:ring-[var(--brand)]";

  const emptyClass =
    "p-3 text-sm text-[var(--text-secondary)] font-jetbrains-mono";

  const itemClass =
    "px-3 py-2 text-sm font-jetbrains-mono text-[var(--text-primary)] cursor-pointer aria-selected:bg-[var(--surface-hover)]";

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={buttonClass}
      >
        {selected ? (
          <span className="block truncate">{selectedLabel}</span>
        ) : (
          <span className={placeholderClass}>{placeholder}</span>
        )}
      </button>

      {open ? (
        <div className={dropdownClass}>
          <Command
            label="Client search"
            className="w-full"
            onKeyDown={(e) => {
              if (e.key === "Escape") setOpen(false);
            }}
          >
            <div className="p-2 border-b border-[var(--border-color)]">
              <Command.Input
                value={query}
                onValueChange={setQuery}
                placeholder="Search client…"
                className={inputClass}
              />
            </div>

            <Command.List className="max-h-56 overflow-auto">
              <Command.Empty>
                <div className={emptyClass}>No results</div>
              </Command.Empty>

              {filtered.map((c) => {
                const emailPart = c.email ? ` • ${c.email}` : "";
                const phonePart = c.phone ? ` • ${c.phone}` : "";
                const label = `${c.full_name}${emailPart}${phonePart}`;

                return (
                  <Command.Item
                    key={c.id}
                    value={label}
                    onSelect={() => {
                      onChange(String(c.id));
                      setOpen(false);
                      setQuery("");
                    }}
                    className={itemClass}
                  >
                    <div className="truncate">{label}</div>
                  </Command.Item>
                );
              })}
            </Command.List>
          </Command>
        </div>
      ) : null}
    </div>
  );
}
