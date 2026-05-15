import { Image as ImageIcon, Plus, Trash2 } from "lucide-react";
import { useCallback, useMemo, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import useUpload from "@/utils/useUpload";
import ImageCarouselModal from "@/components/ImageCarouselModal";

export function PropertyPhotos({ photos, propertyId, userId }) {
  const queryClient = useQueryClient();
  const fileInputRef = useRef(null);

  const [error, setError] = useState(null);
  const [upload, { loading: uploading }] = useUpload();
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const [lightboxIndex, setLightboxIndex] = useState(0);

  const sorted = useMemo(() => {
    const arr = Array.isArray(photos) ? photos : [];
    return [...arr].sort(
      (a, b) => Number(a.sort_order || 0) - Number(b.sort_order || 0),
    );
  }, [photos]);

  const lightboxItems = useMemo(() => {
    return sorted.map((p) => ({
      key: String(p.id),
      url: p.storage_path,
      thumbnailUrl: p.storage_path,
      alt: "Property photo",
    }));
  }, [sorted]);

  const addPhotosMutation = useMutation({
    mutationFn: async (files) => {
      const arr = Array.isArray(files) ? files : [];
      if (!propertyId) {
        throw new Error("Property id missing");
      }

      const uploadedUrls = [];
      for (const file of arr) {
        const { url, error: uploadError } = await upload({ file });
        if (uploadError) {
          throw new Error(uploadError);
        }
        uploadedUrls.push(url);
      }

      const res = await fetch(`/api/properties/${propertyId}/photos`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ photoUrls: uploadedUrls }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not add photos");
      }

      return res.json();
    },
    onSuccess: async () => {
      setError(null);
      await queryClient.invalidateQueries({
        queryKey: ["property", userId, propertyId],
      });
    },
    onError: (e) => {
      console.error(e);
      setError(e?.message || "Could not add photos");
    },
  });

  const deletePhotoMutation = useMutation({
    mutationFn: async (photoId) => {
      const res = await fetch(
        `/api/properties/${propertyId}/photos/${photoId}`,
        {
          method: "DELETE",
        },
      );

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not delete photo");
      }

      return res.json();
    },
    onSuccess: async () => {
      setError(null);
      await queryClient.invalidateQueries({
        queryKey: ["property", userId, propertyId],
      });
    },
    onError: (e) => {
      console.error(e);
      setError(e?.message || "Could not delete photo");
    },
  });

  const onPick = useCallback(() => {
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
      fileInputRef.current.click();
    }
  }, []);

  const onFileChange = useCallback(
    (e) => {
      const files = Array.from(e.target.files || []);
      if (files.length === 0) return;
      setError(null);
      addPhotosMutation.mutate(files);
    },
    [addPhotosMutation],
  );

  const canEdit = !!propertyId;

  return (
    <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6">
      <div className="flex items-center justify-between gap-3 mb-4">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          Photos
        </h2>

        {canEdit ? (
          <>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              className="hidden"
              onChange={onFileChange}
            />
            <button
              type="button"
              onClick={onPick}
              disabled={uploading || addPhotosMutation.isPending}
              className="inline-flex items-center justify-center gap-2 rounded-lg bg-[var(--brand90)] hover:bg-[var(--brand)] text-white px-3 py-2 text-sm font-medium font-jetbrains-mono disabled:opacity-50"
            >
              <Plus size={16} />
              Add photo
            </button>
          </>
        ) : null}
      </div>

      {error ? (
        <div className="mb-4 rounded-lg bg-red-900/15 dark:bg-red-900/30 p-3 text-sm text-red-700 dark:text-red-200 font-jetbrains-mono border border-red-500/20">
          {error}
        </div>
      ) : null}

      {sorted.length === 0 ? (
        <div className="rounded-lg border border-dashed border-gray-300 dark:border-gray-700 p-10 text-center">
          <ImageIcon className="mx-auto mb-3 text-gray-400" size={40} />
          <p className="text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
            No photos yet.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {sorted.map((p, idx) => {
            const deleting =
              deletePhotoMutation.isPending &&
              deletePhotoMutation.variables === p.id;
            return (
              <button
                key={p.id}
                type="button"
                onClick={() => {
                  setLightboxIndex(idx);
                  setLightboxOpen(true);
                }}
                className="relative rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 text-left"
                aria-label="Open photo"
                title="Open"
              >
                <img
                  src={p.storage_path}
                  alt="Property"
                  className="w-full h-28 object-cover"
                />

                {canEdit ? (
                  <button
                    type="button"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      setError(null);
                      deletePhotoMutation.mutate(p.id);
                    }}
                    disabled={deletePhotoMutation.isPending}
                    className="absolute top-2 right-2 inline-flex items-center justify-center h-9 w-9 rounded-full bg-black/60 hover:bg-black/70 text-white disabled:opacity-50"
                    aria-label="Delete photo"
                    title="Delete"
                  >
                    <Trash2 size={16} />
                  </button>
                ) : null}

                {deleting ? (
                  <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
                    <div className="text-xs text-white font-jetbrains-mono">
                      Deleting…
                    </div>
                  </div>
                ) : null}
              </button>
            );
          })}
        </div>
      )}

      <ImageCarouselModal
        open={lightboxOpen}
        title="Photos"
        onClose={() => setLightboxOpen(false)}
        items={lightboxItems}
        initialIndex={lightboxIndex}
        showThumbnails
      />
    </div>
  );
}
