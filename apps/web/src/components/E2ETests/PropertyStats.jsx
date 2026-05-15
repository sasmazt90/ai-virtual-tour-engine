export function PropertyStats({
  photoCount,
  customAssetCount,
  creditsBalance,
}) {
  return (
    <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3">
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-3">
        <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
          Photos
        </div>
        <div className="mt-1 text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          {photoCount}
        </div>
      </div>
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-3">
        <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
          Custom assets
        </div>
        <div className="mt-1 text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          {customAssetCount}
        </div>
      </div>
      <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-3">
        <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
          Credits balance
        </div>
        <div className="mt-1 text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          {Number(creditsBalance || 0).toLocaleString()}
        </div>
      </div>
    </div>
  );
}
