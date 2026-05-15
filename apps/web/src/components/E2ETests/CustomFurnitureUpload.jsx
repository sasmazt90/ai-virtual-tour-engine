export function CustomFurnitureUpload({
  furnitureFile,
  onFileChange,
  disabled,
  uploading,
}) {
  return (
    <div className="mt-4 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
      <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
        Custom furniture for Test 3 (optional)
      </div>
      <div className="mt-1 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
        If your selected property has no custom assets yet, upload one furniture
        image here so Test 3 can run.
      </div>
      <div className="mt-3 flex flex-col sm:flex-row sm:items-center gap-3">
        <input
          type="file"
          accept="image/*"
          disabled={disabled || uploading}
          onChange={(e) => {
            const f = e?.target?.files?.[0] || null;
            onFileChange(f);
          }}
          className="block w-full text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono"
        />
        <div className="text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
          {furnitureFile ? `Selected: ${furnitureFile.name}` : ""}
        </div>
      </div>
    </div>
  );
}
