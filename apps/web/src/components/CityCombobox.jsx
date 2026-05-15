import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, Check } from "lucide-react";

function safeString(v) {
  if (v === null || v === undefined) return "";
  return String(v);
}

function normalize(s) {
  const raw = safeString(s).trim().toLowerCase();
  if (!raw) return "";

  // Tolerant search: remove diacritics and normalize common Turkish chars
  // so "Munchen" matches "München", and "Sanliurfa" matches "Şanlıurfa".
  const turkishFixed = raw
    .replace(/ı/g, "i")
    .replace(/İ/g, "i")
    .replace(/ş/g, "s")
    .replace(/ğ/g, "g")
    .replace(/ü/g, "u")
    .replace(/ö/g, "o")
    .replace(/ç/g, "c");

  return turkishFixed
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

export default function CityCombobox({
  value,
  onChange,
  options,
  placeholder,
  disabled,
}) {
  const wrapperRef = useRef(null);
  const [open, setOpen] = useState(false);

  const resolvedPlaceholder = placeholder || "Start typing…";
  const resolvedOptions = Array.isArray(options) ? options : [];

  const isHugeList = resolvedOptions.length > 8000;

  const query = safeString(value);
  const q = normalize(query);

  const filtered = useMemo(() => {
    const maxResults = 60;

    if (!q) {
      return resolvedOptions.slice(0, maxResults);
    }

    // For extremely large city lists, wait until the user types 2+ chars.
    // This avoids scanning tens of thousands of rows on a single keypress.
    if (isHugeList && q.length < 2) {
      return resolvedOptions.slice(0, maxResults);
    }

    const startsWith = [];
    const includes = [];

    for (const opt of resolvedOptions) {
      const optNorm = normalize(opt);
      if (!optNorm) continue;

      if (optNorm.startsWith(q)) {
        startsWith.push(opt);
      } else if (optNorm.includes(q)) {
        includes.push(opt);
      }

      if (startsWith.length + includes.length >= maxResults) {
        break;
      }
    }

    return startsWith.concat(includes).slice(0, maxResults);
  }, [isHugeList, q, resolvedOptions]);

  const footerText = useMemo(() => {
    if (isHugeList && q && q.length < 2) {
      return "Type 2+ letters to search • Showing first results";
    }
    return "Type to filter • Showing up to 60 results";
  }, [isHugeList, q]);

  useEffect(() => {
    function onDocMouseDown(e) {
      const el = wrapperRef.current;
      if (!el) return;
      if (el.contains(e.target)) return;
      setOpen(false);
    }

    if (open) {
      document.addEventListener("mousedown", onDocMouseDown);
      return () => document.removeEventListener("mousedown", onDocMouseDown);
    }
  }, [open]);

  return (
    <div ref={wrapperRef} className="relative">
      <input
        value={query}
        disabled={!!disabled}
        onChange={(e) => {
          const next = e.target.value;
          onChange(next);
          if (!open) setOpen(true);
        }}
        onFocus={() => {
          if (!disabled) setOpen(true);
        }}
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            setOpen(false);
          }
        }}
        placeholder={resolvedPlaceholder}
        className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono disabled:opacity-60 pr-10"
        autoComplete="off"
      />

      <div className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none">
        <ChevronDown size={16} />
      </div>

      {open && !disabled ? (
        <div className="absolute z-30 mt-2 w-full rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-[#1E1E1E] shadow-xl overflow-hidden">
          <div className="max-h-60 overflow-auto">
            {filtered.length === 0 ? (
              <div className="p-3 text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                No matches
              </div>
            ) : (
              filtered.map((opt) => {
                const isSelected = opt === query;
                return (
                  <button
                    key={opt}
                    type="button"
                    onClick={() => {
                      onChange(opt);
                      setOpen(false);
                    }}
                    className="w-full text-left px-4 py-2 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="min-w-0 truncate text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                        {opt}
                      </div>
                      {isSelected ? (
                        <Check size={16} className="text-[var(--brand)]" />
                      ) : null}
                    </div>
                  </button>
                );
              })
            )}
          </div>

          <div className="border-t border-gray-200 dark:border-gray-700 px-3 py-2 text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
            {footerText}
          </div>
        </div>
      ) : null}
    </div>
  );
}
