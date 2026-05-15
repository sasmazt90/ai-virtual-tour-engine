export function FilledFieldsDebug({ filledFields }) {
  return (
    <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6">
      <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 font-jetbrains-mono">
        Filled Fields
      </h2>
      <pre className="text-xs text-gray-700 dark:text-gray-200 whitespace-pre-wrap font-jetbrains-mono">
        {JSON.stringify(filledFields || {}, null, 2)}
      </pre>
    </div>
  );
}
