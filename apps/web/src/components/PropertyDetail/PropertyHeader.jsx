import { ArrowLeft, Link as LinkIcon, Pencil, Trash2 } from "lucide-react";
import { useCallback, useState } from "react";

export function PropertyHeader({ property, propertyId, onOpenShare }) {
  const [deleting, setDeleting] = useState(false);

  const onDelete = useCallback(async () => {
    const ok =
      typeof window !== "undefined"
        ? window.confirm("Delete this property? This cannot be undone.")
        : false;
    if (!ok) return;

    setDeleting(true);
    try {
      const res = await fetch(`/api/properties/${propertyId}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not delete property");
      }
      if (typeof window !== "undefined") {
        window.location.href = "/properties";
      }
    } catch (e) {
      console.error(e);
      if (typeof window !== "undefined") {
        window.alert("Could not delete this property.");
      }
      setDeleting(false);
    }
  }, [propertyId]);

  return (
    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-8">
      <div>
        <a
          href="/properties"
          className="inline-flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 font-jetbrains-mono"
        >
          <ArrowLeft size={16} />
          Back to Properties
        </a>
        <h1 className="mt-2 text-3xl sm:text-4xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          {property.title}
        </h1>
        <div className="mt-2 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
          {property.address_line
            ? `${property.address_line}${property.city ? ", " + property.city : ""}`
            : "No address"}
        </div>
      </div>

      <div className="flex flex-col sm:flex-row gap-3">
        <button
          type="button"
          onClick={onOpenShare}
          className="inline-flex items-center justify-center gap-2 px-5 py-3 bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 rounded-lg font-medium transition-colors hover:bg-gray-800 dark:hover:bg-gray-200 font-jetbrains-mono"
        >
          <LinkIcon size={18} />
          Share with client
        </button>

        <a
          href={`/properties/${propertyId}/edit`}
          className="inline-flex items-center justify-center gap-2 px-5 py-3 bg-white dark:bg-[#262626] text-gray-900 dark:text-gray-100 rounded-lg font-medium transition-colors hover:bg-gray-50 dark:hover:bg-gray-800 font-jetbrains-mono border border-gray-200 dark:border-gray-700"
        >
          <Pencil size={18} />
          Edit
        </a>

        <button
          type="button"
          onClick={onDelete}
          disabled={deleting}
          className="inline-flex items-center justify-center gap-2 px-5 py-3 bg-white dark:bg-[#262626] text-red-700 dark:text-red-300 rounded-lg font-medium transition-colors hover:bg-red-50 dark:hover:bg-red-900/20 font-jetbrains-mono border border-red-200/70 dark:border-red-900/50 disabled:opacity-50"
        >
          <Trash2 size={18} />
          {deleting ? "Deleting…" : "Delete"}
        </button>
      </div>
    </div>
  );
}
