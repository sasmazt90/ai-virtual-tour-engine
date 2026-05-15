export function TogglePill({ label, selected, onClick }) {
  const cls = selected
    ? "bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900"
    : "bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700";

  return (
    <button
      type="button"
      onClick={onClick}
      className={`px-4 py-2 rounded-lg font-medium transition-colors font-jetbrains-mono ${cls}`}
    >
      {label}
    </button>
  );
}
