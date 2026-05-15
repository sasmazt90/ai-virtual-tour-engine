import { ModalShell } from "./ModalShell";
import { ClientCombobox } from "./ClientCombobox";

export function CreateEventModal({
  open,
  onClose,
  mode,
  onDelete,
  isDeleting,
  formDate,
  setFormDate,
  selectedDate,
  formClientId,
  setFormClientId,
  formType,
  setFormType,
  formTime,
  setFormTime,
  formDuration,
  setFormDuration,
  formPropertyId,
  setFormPropertyId,
  formAddress,
  setFormAddress,
  formNotes,
  setFormNotes,
  formError,
  clients,
  properties,
  onSubmit,
  isSubmitting,
}) {
  if (!open) return null;

  const isEdit = mode === "edit";
  const title = isEdit ? "Edit meeting" : "Add meeting";
  const submitLabel = isEdit ? "Update" : "Save";

  // Keep meeting type values and labels in English.
  const meetingTypeOptions = [
    { value: "phone", label: "Phone call" },
    { value: "email", label: "Email" },
    { value: "property_visit", label: "Property visit" },
    { value: "in_person", label: "In-person meeting" },
  ];

  const durationOptions = [
    { value: "30", label: "30 min" },
    { value: "60", label: "60 min" },
    { value: "90", label: "90 min" },
  ];

  // IMPORTANT: Use ThemeProvider CSS variables so inputs stay readable.
  const inputClass =
    "mt-1 w-full rounded-lg border border-[var(--border-color)] bg-[var(--card-bg)] px-3 py-2 text-sm font-jetbrains-mono text-[var(--text-primary)] placeholder-[var(--text-secondary)] outline-none focus:ring-2 focus:ring-[var(--brand)]";

  // Use the same base style for selects, but give them a slightly clearer surface.
  const selectClass =
    "mt-1 w-full rounded-lg border border-[var(--border-color)] bg-[var(--surface-muted)] px-3 py-2 text-sm font-jetbrains-mono text-[var(--text-primary)] outline-none focus:ring-2 focus:ring-[var(--brand)]";

  const labelClass = "text-sm font-jetbrains-mono text-[var(--text-secondary)]";

  const cancelButtonClass =
    "px-4 py-2 rounded-lg border border-[var(--border-color)] bg-[var(--card-bg)] hover:bg-black/5 text-[var(--text-primary)] font-jetbrains-mono transition-colors";

  return (
    <ModalShell title={title} onClose={onClose}>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="sm:col-span-2">
          <div className={labelClass}>Date</div>
          <input
            type="date"
            value={formDate}
            onChange={(e) => setFormDate(e.target.value)}
            className={inputClass}
          />
        </div>

        <div className="sm:col-span-2">
          <div className={labelClass}>Client</div>
          <div className="mt-1">
            <ClientCombobox
              value={formClientId}
              onChange={setFormClientId}
              clients={clients}
              placeholder="Select a client…"
            />
          </div>
        </div>

        <div>
          <div className={labelClass}>Time (HH:MM)</div>
          <input
            value={formTime}
            onChange={(e) => setFormTime(e.target.value)}
            placeholder="09:00"
            className={inputClass}
          />
        </div>

        <div>
          <div className={labelClass}>Duration</div>
          <select
            value={formDuration}
            onChange={(e) => setFormDuration(e.target.value)}
            className={selectClass}
          >
            {durationOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div className="sm:col-span-2">
          <div className={labelClass}>Meeting type</div>
          <select
            value={formType}
            onChange={(e) => setFormType(e.target.value)}
            className={selectClass}
          >
            {meetingTypeOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        {formType === "property_visit" ? (
          <div className="sm:col-span-2">
            <div className={labelClass}>Property</div>
            <select
              value={formPropertyId}
              onChange={(e) => setFormPropertyId(e.target.value)}
              className={selectClass}
            >
              <option value="">Select a property…</option>
              {(Array.isArray(properties) ? properties : []).map((p) => (
                <option key={p.id} value={p.id}>
                  {p.title}
                </option>
              ))}
            </select>
          </div>
        ) : null}

        {formType === "in_person" ? (
          <div className="sm:col-span-2">
            <div className={labelClass}>Address</div>
            <input
              value={formAddress}
              onChange={(e) => setFormAddress(e.target.value)}
              placeholder="Enter address…"
              className={inputClass}
            />
          </div>
        ) : null}

        <div className="sm:col-span-2">
          <div className={labelClass}>Notes</div>
          <textarea
            value={formNotes}
            onChange={(e) => setFormNotes(e.target.value)}
            rows={4}
            placeholder="Add notes…"
            className={inputClass}
          />
        </div>
      </div>

      {formError ? (
        <div className="mt-4 rounded-lg bg-red-900/15 dark:bg-red-900/30 p-3 text-sm text-red-700 dark:text-red-200 font-jetbrains-mono border border-red-500/20">
          {formError}
        </div>
      ) : null}

      <div className="mt-5 flex flex-col sm:flex-row gap-3 justify-end">
        {isEdit ? (
          <button
            type="button"
            onClick={onDelete}
            disabled={isDeleting}
            className="px-4 py-2 rounded-lg border border-red-500/30 bg-red-600/10 hover:bg-red-600/15 text-red-700 dark:text-red-200 font-jetbrains-mono transition-colors disabled:opacity-50"
          >
            {isDeleting ? "Deleting…" : "Delete"}
          </button>
        ) : null}

        <button type="button" onClick={onClose} className={cancelButtonClass}>
          Cancel
        </button>

        <button
          type="button"
          disabled={isSubmitting}
          onClick={onSubmit}
          className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-[var(--brand90)] hover:bg-[var(--brand)] text-white font-medium font-jetbrains-mono transition-colors disabled:opacity-50"
        >
          {isSubmitting ? "Saving…" : submitLabel}
        </button>
      </div>
    </ModalShell>
  );
}
