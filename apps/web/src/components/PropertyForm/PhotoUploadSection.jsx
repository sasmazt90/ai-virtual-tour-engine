import { Camera } from "lucide-react";
import { useCallback, useMemo, useRef, useState } from "react";

export function PhotoUploadSection({ onPickPhotos, photoPreviews }) {
  const inputRef = useRef(null);
  const [fileLabel, setFileLabel] = useState("No files chosen");

  const onChooseClick = useCallback(() => {
    if (inputRef.current) {
      inputRef.current.click();
    }
  }, []);

  const handleChange = useCallback(
    (e) => {
      const files = Array.from(e.target.files || []);
      if (files.length === 0) {
        setFileLabel("No files chosen");
      } else if (files.length === 1) {
        setFileLabel(files[0].name);
      } else {
        setFileLabel(`${files.length} files selected`);
      }

      onPickPhotos(e);
    },
    [onPickPhotos],
  );

  const helpText = useMemo(() => {
    return "JPG/PNG recommended. You can select multiple photos.";
  }, []);

  return (
    <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6 sm:p-8">
      <div className="flex items-center gap-3 mb-6">
        <Camera className="text-[var(--brand)]" />
        <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          Photo Upload
        </h2>
      </div>

      <div className="space-y-4">
        {/* Custom file input so browser locale can't show Turkish labels like "Dosyaları Seç" */}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          multiple
          onChange={handleChange}
          className="hidden"
        />

        <div className="flex flex-col sm:flex-row sm:items-center gap-3">
          <button
            type="button"
            onClick={onChooseClick}
            className="inline-flex items-center justify-center px-4 py-3 rounded-lg bg-[var(--brand90)] hover:bg-[var(--brand)] text-white font-medium transition-colors font-jetbrains-mono"
          >
            Choose files
          </button>
          <div className="text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono break-all">
            {fileLabel}
          </div>
        </div>

        <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
          {helpText}
        </div>

        {photoPreviews.length > 0 ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {photoPreviews.map((src, idx) => (
              <div
                key={`${src}-${idx}`}
                className="relative rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700"
              >
                <img
                  src={src}
                  alt={`Upload ${idx + 1}`}
                  className="w-full h-28 object-cover"
                />
              </div>
            ))}
          </div>
        ) : (
          <div className="rounded-lg border border-dashed border-gray-300 dark:border-gray-700 p-6 text-center">
            <p className="text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
              Add some photos to make the listing look great.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
