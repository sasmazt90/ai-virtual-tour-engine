import { Loader2, Save } from "lucide-react";
import { FieldRow } from "./FieldRow";

export function ContractFieldsEditor({
  schema,
  requiredSet,
  editableSet,
  editFields,
  isSigned,
  onFieldChange,
  onSave,
  isSaving,
  suggestionsMap,
}) {
  return (
    <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          Contract Fields
        </h2>
        {!isSigned ? (
          <button
            type="button"
            onClick={onSave}
            disabled={isSaving}
            className="inline-flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:bg-gray-800 dark:hover:bg-gray-200 disabled:opacity-50 font-jetbrains-mono"
            title="Save changes"
          >
            {isSaving ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <Save size={16} />
            )}
            Save
          </button>
        ) : null}
      </div>

      <div className="mt-2 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
        <span className="text-red-600">*</span> required
      </div>

      {!isSigned ? (
        <div className="mt-2 text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
          Some fields are locked for legal integrity (party + property
          identity).
        </div>
      ) : null}

      {schema.sections.map((sec) => (
        <div key={sec.title} className="mt-5">
          <div className="mb-3">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
              {sec.title}
            </h3>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            {sec.fields.map((key) => {
              const required = requiredSet.has(key);
              const value = editFields?.[key] || "";
              const isEditable = editableSet ? editableSet.has(key) : true;
              const disabled = isSigned || !isEditable;
              const disabledReason =
                !isSigned && !isEditable ? "Locked (immutable field)" : null;

              const suggestions = Array.isArray(suggestionsMap?.[key])
                ? suggestionsMap[key]
                : [];

              return (
                <FieldRow
                  key={key}
                  fieldKey={key}
                  value={value}
                  required={required}
                  disabled={disabled}
                  disabledReason={disabledReason}
                  suggestions={suggestions}
                  onChange={(val) => onFieldChange(key, val)}
                />
              );
            })}
          </div>
        </div>
      ))}

      {isSigned ? (
        <div className="mt-4 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
          Editing is disabled after signing.
        </div>
      ) : null}
    </div>
  );
}
