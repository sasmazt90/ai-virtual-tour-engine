export function PropertySelector({
  selectedPropertyId,
  onPropertyChange,
  properties,
  disabled,
}) {
  const propertiesList = Array.isArray(properties) ? properties : [];

  return (
    <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-5">
      <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
        Choose a property
      </div>

      <div className="mt-3">
        <select
          value={selectedPropertyId}
          onChange={(e) => onPropertyChange(e.target.value)}
          disabled={disabled}
          className="w-full rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-[#1E1E1E] text-gray-900 dark:text-gray-100 px-3 py-2 text-sm font-jetbrains-mono"
        >
          <option value="">Select…</option>
          {propertiesList.map((p) => {
            const title = p?.title ? String(p.title) : "Untitled";
            return (
              <option key={p.id} value={p.id}>
                {title}
              </option>
            );
          })}
        </select>
      </div>
    </div>
  );
}
