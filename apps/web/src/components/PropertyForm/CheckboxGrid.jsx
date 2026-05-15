import { useMemo } from "react";

function isGroupedOptions(options) {
  return (
    Array.isArray(options) &&
    options.length > 0 &&
    typeof options[0] === "object" &&
    options[0] !== null &&
    Array.isArray(options[0].options)
  );
}

export function CheckboxGrid({ options, selected, setSelected }) {
  const selectedSet = useMemo(() => {
    return new Set(Array.isArray(selected) ? selected : []);
  }, [selected]);

  const toggle = (opt, checked) => {
    const next = new Set(selectedSet);
    if (checked) {
      next.add(opt);
    } else {
      next.delete(opt);
    }
    setSelected(Array.from(next));
  };

  const renderOption = (opt) => {
    const checked = selectedSet.has(opt);
    return (
      <label
        key={opt}
        className="flex items-center gap-2 rounded-lg border border-gray-200 dark:border-gray-700 px-3 py-2 bg-white dark:bg-[#1E1E1E]"
      >
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => toggle(opt, e.target.checked)}
        />
        <span className="text-sm text-gray-800 dark:text-gray-200 font-jetbrains-mono">
          {opt}
        </span>
      </label>
    );
  };

  if (isGroupedOptions(options)) {
    return (
      <div className="space-y-5">
        {options.map((group) => {
          const groupTitle = group.title || "";
          const groupOptions = Array.isArray(group.options)
            ? group.options
            : [];

          return (
            <div key={groupTitle || Math.random()}>
              <div className="text-xs uppercase tracking-wide text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                {groupTitle}
              </div>
              <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-2">
                {groupOptions.map((opt) => renderOption(opt))}
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
      {Array.isArray(options) ? options.map((opt) => renderOption(opt)) : null}
    </div>
  );
}
