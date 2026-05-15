import { useEffect, useMemo, useRef, useState } from "react";
import { FIELD_META } from "@/utils/contractSchema";

export function FieldRow({
  fieldKey,
  value,
  required,
  disabled,
  disabledReason,
  onChange,
  suggestions,
}) {
  const meta = FIELD_META[fieldKey] || { label: fieldKey, type: "text" };
  const isTextarea = meta.type === "textarea";
  const labelText = meta.label;

  const wrapperClassName =
    fieldKey === "propertyAddress" ||
    fieldKey === "additionalTerms" ||
    fieldKey === "PROPERTY_ADDRESS" ||
    fieldKey === "OWNER_ADDRESS" ||
    fieldKey === "CUSTOMER_ADDRESS"
      ? "space-y-2 sm:col-span-2"
      : "space-y-2";

  const requiredMark = required ? (
    <span className="text-red-600">*</span>
  ) : null;

  const inputClass =
    "w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono";

  const disabledClass = disabled ? " opacity-70 cursor-not-allowed" : "";

  const [focused, setFocused] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const closeTimer = useRef(null);

  const normalizedSuggestions = useMemo(() => {
    const raw = Array.isArray(suggestions) ? suggestions : [];
    const out = [];
    const seen = new Set();
    for (const s of raw) {
      if (typeof s !== "string") continue;
      const v = s.trim();
      if (!v) continue;
      if (seen.has(v.toLowerCase())) continue;
      seen.add(v.toLowerCase());
      out.push(v);
    }
    return out;
  }, [suggestions]);

  const filteredSuggestions = useMemo(() => {
    const q =
      typeof value === "string" ? value.trim() : String(value || "").trim();
    if (q.length < 3) return [];

    const ql = q.toLowerCase();
    return normalizedSuggestions
      .filter((s) => s.toLowerCase().includes(ql))
      .slice(0, 8);
  }, [normalizedSuggestions, value]);

  const shouldShowDropdown =
    !disabled &&
    focused &&
    filteredSuggestions.length > 0 &&
    (typeof value === "string"
      ? value.trim().length
      : String(value || "").trim().length) >= 3;

  useEffect(() => {
    if (shouldShowDropdown) {
      setDropdownOpen(true);
    } else {
      setDropdownOpen(false);
    }
  }, [shouldShowDropdown]);

  useEffect(() => {
    return () => {
      if (closeTimer.current) {
        clearTimeout(closeTimer.current);
      }
    };
  }, []);

  const handleBlur = () => {
    // Delay close so a click on a suggestion can register.
    if (closeTimer.current) {
      clearTimeout(closeTimer.current);
    }
    closeTimer.current = setTimeout(() => {
      setFocused(false);
      setDropdownOpen(false);
    }, 120);
  };

  const handleFocus = () => {
    if (closeTimer.current) {
      clearTimeout(closeTimer.current);
    }
    setFocused(true);
  };

  return (
    <div className={wrapperClassName}>
      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
        {labelText} {requiredMark}
      </label>
      {disabled && disabledReason ? (
        <div className="text-[11px] text-gray-500 dark:text-gray-400 font-jetbrains-mono">
          {disabledReason}
        </div>
      ) : null}

      <div className="relative">
        {isTextarea ? (
          <textarea
            value={value}
            onChange={(e) => onChange(e.target.value)}
            rows={6}
            disabled={disabled}
            className={inputClass + disabledClass}
            placeholder={meta.placeholder || ""}
            onFocus={handleFocus}
            onBlur={handleBlur}
          />
        ) : (
          <input
            type={meta.type || "text"}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            disabled={disabled}
            className={inputClass + disabledClass}
            placeholder={meta.placeholder || ""}
            onFocus={handleFocus}
            onBlur={handleBlur}
            autoComplete="off"
          />
        )}

        {dropdownOpen ? (
          <div className="absolute z-20 mt-2 w-full rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-[#1f1f1f] shadow-lg overflow-hidden">
            {filteredSuggestions.map((opt) => (
              <button
                key={opt}
                type="button"
                className="w-full text-left px-3 py-2 text-sm text-gray-900 dark:text-gray-100 hover:bg-gray-50 dark:hover:bg-gray-800 font-jetbrains-mono"
                onMouseDown={(e) => {
                  // Prevent blur before click.
                  e.preventDefault();
                }}
                onClick={() => {
                  onChange(opt);
                  setDropdownOpen(false);
                }}
              >
                {opt}
              </button>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}
