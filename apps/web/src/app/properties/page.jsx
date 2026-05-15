import { Header } from "../../components/Header";
import { useState, useCallback, useRef } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import useUser from "@/utils/useUser";
import useUpload from "@/utils/useUpload";
import {
  Plus,
  Home,
  Banknote,
  MapPin,
  Trash2,
  Image as ImageIcon,
} from "lucide-react";

function getCoverPhoto(property) {
  const arr = Array.isArray(property?.photos) ? property.photos : [];
  const sorted = [...arr].sort(
    (a, b) => Number(a.sort_order || 0) - Number(b.sort_order || 0),
  );
  return sorted[0] || null;
}

function PropertyCard({ property, userId, onInvalidate }) {
  const queryClient = useQueryClient();
  const fileInputRef = useRef(null);
  const [error, setError] = useState(null);

  const [upload, { loading: uploading }] = useUpload();

  const cover = getCoverPhoto(property);

  const addPhotosMutation = useMutation({
    mutationFn: async (files) => {
      setError(null);
      const arr = Array.isArray(files) ? files : [];
      if (arr.length === 0) return;

      const uploadedUrls = [];
      for (const file of arr) {
        const { url, error: upErr } = await upload({ file });
        if (upErr) throw new Error(upErr);
        uploadedUrls.push(url);
      }

      const res = await fetch(`/api/properties/${property.id}/photos`, {
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
      await queryClient.invalidateQueries({ queryKey: ["properties", userId] });
      await onInvalidate?.();
    },
    onError: (e) => {
      console.error(e);
      setError(e?.message || "Could not add photos");
    },
  });

  const deleteCoverMutation = useMutation({
    mutationFn: async () => {
      if (!cover?.id) throw new Error("No photo to delete");
      const res = await fetch(
        `/api/properties/${property.id}/photos/${cover.id}`,
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
      await queryClient.invalidateQueries({ queryKey: ["properties", userId] });
      await onInvalidate?.();
    },
    onError: (e) => {
      console.error(e);
      setError(e?.message || "Could not delete photo");
    },
  });

  const isBusy =
    uploading || addPhotosMutation.isPending || deleteCoverMutation.isPending;

  const onPick = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
      fileInputRef.current.click();
    }
  }, []);

  const onDeleteCover = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      setError(null);
      deleteCoverMutation.mutate();
    },
    [deleteCoverMutation],
  );

  return (
    <a
      href={`/properties/${property.id}`}
      className="bg-white/70 dark:bg-white/5 rounded-xl border border-black/10 dark:border-white/10 backdrop-blur shadow-[0_14px_60px_rgba(0,0,0,0.18)] overflow-hidden hover:bg-white/80 dark:hover:bg-white/10 transition-colors"
    >
      <div className="relative">
        {cover ? (
          <img
            src={cover.storage_path}
            alt={property.title}
            className="w-full h-48 object-cover"
          />
        ) : (
          <div className="w-full h-48 bg-gray-200 dark:bg-gray-700 flex items-center justify-center">
            <Home className="text-gray-400" size={48} />
          </div>
        )}

        {/* hidden file input per card */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={(e) => {
            const files = Array.from(e.target.files || []);
            if (files.length === 0) return;
            addPhotosMutation.mutate(files);
          }}
        />

        {/* photo actions */}
        <div className="absolute top-3 right-3 flex items-center gap-2">
          <button
            type="button"
            onClick={onPick}
            disabled={isBusy}
            className="inline-flex items-center justify-center h-9 w-9 rounded-full bg-black/60 hover:bg-black/70 text-white disabled:opacity-50"
            title="Add photos"
            aria-label="Add photos"
          >
            <Plus size={16} />
          </button>

          {cover ? (
            <button
              type="button"
              onClick={onDeleteCover}
              disabled={isBusy}
              className="inline-flex items-center justify-center h-9 w-9 rounded-full bg-black/60 hover:bg-black/70 text-white disabled:opacity-50"
              title="Delete cover photo"
              aria-label="Delete cover photo"
            >
              <Trash2 size={16} />
            </button>
          ) : null}
        </div>

        {isBusy ? (
          <div className="absolute inset-0 bg-black/35 flex items-center justify-center">
            <div className="text-xs text-white font-jetbrains-mono">
              Working…
            </div>
          </div>
        ) : null}
      </div>

      <div className="p-6">
        {error ? (
          <div className="mb-3 rounded-lg bg-red-50 dark:bg-red-900/30 p-3 text-sm text-red-600 dark:text-red-300 font-jetbrains-mono">
            {error}
          </div>
        ) : null}

        <div className="flex items-start justify-between mb-3">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            {property.title}
          </h3>
          <span
            className={`px-2 py-1 text-xs rounded-full font-medium ${
              property.property_status === "for_sale"
                ? "bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200"
                : "bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200"
            }`}
          >
            {property.property_status === "for_sale" ? "For Sale" : "For Rent"}
          </span>
        </div>

        <div className="space-y-2">
          {property.price && (
            <div className="flex items-center text-gray-700 dark:text-gray-300 font-jetbrains-mono">
              <Banknote size={16} className="mr-2 text-[var(--brand)]" />
              <span className="text-lg font-semibold">
                {parseFloat(property.price).toLocaleString()}
                {property?.currency ? ` ${String(property.currency)}` : ""}
              </span>
            </div>
          )}
          {property.address_line && (
            <div className="flex items-start text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
              <MapPin
                size={16}
                className="mr-2 mt-0.5 flex-shrink-0 text-[var(--brand)]"
              />
              <span>
                {property.address_line}, {property.city}
              </span>
            </div>
          )}
          {property.rooms && (
            <p className="text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
              {property.rooms} rooms • {property.size_sqm} m²
            </p>
          )}

          <div className="pt-2 flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400 font-jetbrains-mono">
            <ImageIcon size={14} className="text-[var(--brand)]" />
            <span>
              {Array.isArray(property?.photos) ? property.photos.length : 0}{" "}
              photos
            </span>
          </div>
        </div>
      </div>
    </a>
  );
}

export default function PropertiesPage() {
  const { data: user, loading: userLoading } = useUser();
  const [statusFilter, setStatusFilter] = useState("all");
  const [search, setSearch] = useState("");

  const {
    data: properties,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: ["properties", user?.id, statusFilter, search],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (statusFilter !== "all") params.append("status", statusFilter);
      if (search) params.append("search", search);

      const res = await fetch(`/api/properties?${params.toString()}`);
      if (!res.ok) throw new Error("Failed to fetch properties");
      return res.json();
    },
    enabled: !!user?.id,
  });

  if (userLoading) {
    return (
      <div className="min-h-screen ui-surface">
        <Header />
        <div className="pt-16 flex items-center justify-center min-h-screen">
          <p className="text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Loading...
          </p>
        </div>
      </div>
    );
  }

  if (!user) {
    if (typeof window !== "undefined") {
      window.location.href = "/account/signin";
    }
    return null;
  }

  return (
    <div className="min-h-screen ui-surface">
      <Header />

      <div className="pt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-8 py-8 sm:py-12">
          <div className="mb-8">
            <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-gray-100 mb-2 font-jetbrains-mono">
              Properties
            </h1>
            <p className="text-gray-600 dark:text-gray-300 font-jetbrains-mono">
              Manage your property listings
            </p>
          </div>

          <div className="mb-8 flex flex-col sm:flex-row gap-4">
            <div className="flex-1">
              <input
                type="text"
                placeholder="Search properties..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full px-4 py-3 bg-white dark:bg-[#262626] border border-gray-200 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
              />
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setStatusFilter("all")}
                className={`px-4 py-2 rounded-lg font-medium transition-colors font-jetbrains-mono ${
                  statusFilter === "all"
                    ? "bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900"
                    : "bg-white dark:bg-[#262626] text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-700"
                }`}
              >
                All
              </button>
              <button
                onClick={() => setStatusFilter("for_sale")}
                className={`px-4 py-2 rounded-lg font-medium transition-colors font-jetbrains-mono ${
                  statusFilter === "for_sale"
                    ? "bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900"
                    : "bg-white dark:bg-[#262626] text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-700"
                }`}
              >
                For Sale
              </button>
              <button
                onClick={() => setStatusFilter("for_rent")}
                className={`px-4 py-2 rounded-lg font-medium transition-colors font-jetbrains-mono ${
                  statusFilter === "for_rent"
                    ? "bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900"
                    : "bg-white dark:bg-[#262626] text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-700"
                }`}
              >
                For Rent
              </button>
            </div>
          </div>

          {isLoading ? (
            <div className="text-center py-12">
              <p className="text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                Loading properties...
              </p>
            </div>
          ) : properties && properties.length > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {properties.map((property) => (
                <PropertyCard
                  key={property.id}
                  property={property}
                  userId={user?.id}
                  onInvalidate={refetch}
                />
              ))}
            </div>
          ) : (
            <div className="text-center py-12 bg-white/70 dark:bg-white/5 border border-black/10 dark:border-white/10 rounded-xl backdrop-blur">
              <Home className="mx-auto mb-4 text-gray-400" size={48} />
              <p className="text-gray-600 dark:text-gray-400 mb-4 font-jetbrains-mono">
                No properties yet
              </p>
              <a
                href="/properties/new"
                className="inline-flex items-center px-6 py-3 bg-[var(--brand90)] hover:bg-[var(--brand)] text-white rounded-lg font-medium transition-colors font-jetbrains-mono"
              >
                <Plus size={20} className="mr-2" />
                Add Your First Property
              </a>
            </div>
          )}
        </div>
      </div>

      {properties && properties.length > 0 && (
        <a
          href="/properties/new"
          className="fixed bottom-8 right-8 w-14 h-14 bg-[var(--brand)] hover:bg-[var(--brandHover)] text-white rounded-full shadow-lg flex items-center justify-center transition-colors"
        >
          <Plus size={24} />
        </a>
      )}
    </div>
  );
}
