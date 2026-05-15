import { useMemo, useState, useEffect, useCallback } from "react";
import {
  Image as ImageIcon,
  Images,
  Flashlight,
  FlashlightOff,
  Moon,
  Sun,
  Trash2,
} from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import ImageCarouselModal from "@/components/ImageCarouselModal";

function safeNumber(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function readMetaNumber(meta, key) {
  if (!meta || typeof meta !== "object") return null;
  return safeNumber(meta?.[key]);
}

function variantKey({ isNight, isLightOn }) {
  return `${isNight ? "night" : "day"}_light_${isLightOn ? "on" : "off"}`;
}

function pickVariantUrl(variants, { isNight, isLightOn }) {
  if (!variants || typeof variants !== "object") return "";

  const desiredKey = variantKey({ isNight, isLightOn });
  const otherTimeSameLightKey = variantKey({ isNight: !isNight, isLightOn });
  const sameTimeOtherLightKey = variantKey({ isNight, isLightOn: !isLightOn });
  const otherTimeOtherLightKey = variantKey({
    isNight: !isNight,
    isLightOn: !isLightOn,
  });

  const keys = [
    desiredKey,
    otherTimeSameLightKey,
    sameTimeOtherLightKey,
    otherTimeOtherLightKey,
  ];

  for (const k of keys) {
    const v = variants[k];
    const url =
      (v &&
        typeof v === "object" &&
        (v.storage_path || v.url || v.download_url)) ||
      (typeof v === "string" ? v : "");
    if (typeof url === "string" && url.trim().length > 0) {
      return url.trim();
    }
  }

  return "";
}

function StagingCard({ title, thumbUrl, photoCount, onOpen }) {
  return (
    <button
      type="button"
      onClick={onOpen}
      className="rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 text-left hover:bg-gray-50/60 dark:hover:bg-gray-800/40"
      aria-label="Open staging"
      title="Open"
    >
      <div className="relative">
        {thumbUrl ? (
          <img
            src={thumbUrl}
            alt="Staging thumbnail"
            className="w-full h-36 object-cover"
            loading="lazy"
          />
        ) : (
          <div className="w-full h-36 flex items-center justify-center text-gray-400">
            <ImageIcon size={28} />
          </div>
        )}

        {Number(photoCount || 0) > 1 ? (
          <div className="absolute bottom-2 right-2 inline-flex items-center gap-1 rounded-full bg-white/90 dark:bg-gray-900/90 border border-gray-200 dark:border-gray-700 px-2 py-1 text-xs text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            <Images size={14} />
            <span>{Number(photoCount || 0)}</span>
          </div>
        ) : null}
      </div>

      <div className="px-3 py-2">
        <div className="text-xs text-gray-700 dark:text-gray-200 font-jetbrains-mono">
          {title}
        </div>
        <div className="mt-1 text-[11px] text-gray-500 dark:text-gray-400 font-jetbrains-mono">
          Click to preview
        </div>
      </div>
    </button>
  );
}

export function StagingsSection({
  stagings,
  formatStagingLabel,
  onAddNew,
  onRefresh,
}) {
  const addNewButtonClass =
    "inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-200 text-sm font-medium hover:bg-gray-50 dark:hover:bg-gray-800 font-jetbrains-mono";

  const cards = useMemo(() => {
    const out = [];
    const list = Array.isArray(stagings) ? stagings : [];

    for (const s of list) {
      const title = formatStagingLabel(s);
      const metaStaged = Array.isArray(s?.meta?.staged) ? s.meta.staged : [];

      if (metaStaged.length > 0) {
        const firstVariants =
          metaStaged?.[0]?.variants &&
          typeof metaStaged[0].variants === "object"
            ? metaStaged[0].variants
            : null;

        const thumbUrl = firstVariants
          ? pickVariantUrl(firstVariants, { isNight: false, isLightOn: false })
          : "";

        out.push({
          key: String(s.id),
          staging: s,
          title,
          photoCount: metaStaged.length,
          hasVariants: true,
          thumbUrl,
        });
        continue;
      }

      // Old format: fall back to staging_images list
      const images = Array.isArray(s?.images) ? s.images : [];
      const urls = images
        .map((i) => i?.storage_path)
        .filter((u) => typeof u === "string" && u.trim().length > 0)
        .map((u) => u.trim());

      out.push({
        key: String(s.id),
        staging: s,
        title,
        photoCount: urls.length,
        hasVariants: false,
        thumbUrl: urls[0] || "",
      });
    }

    return out;
  }, [formatStagingLabel, stagings]);

  const [activeCard, setActiveCard] = useState(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [isNight, setIsNight] = useState(false);
  const [isLightOn, setIsLightOn] = useState(false);
  const [deleteError, setDeleteError] = useState(null);

  const isVacantActive = activeCard?.staging?.staging_type === "vacant";

  useEffect(() => {
    if (!activeCard) return;
    setIsNight(false);
    setIsLightOn(false);
    setActiveIndex(0);
    setDeleteError(null);
  }, [activeCard]);

  const deleteStagingMutation = useMutation({
    mutationFn: async ({ stagingId }) => {
      const res = await fetch(
        `/api/stagings/${encodeURIComponent(stagingId)}`,
        {
          method: "DELETE",
        },
      );

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not delete staging");
      }

      return res.json();
    },
    onSuccess: async () => {
      setActiveCard(null);
      if (typeof onRefresh === "function") {
        await onRefresh();
      }
    },
    onError: (err) => {
      console.error(err);
      setDeleteError(err?.message || "Could not delete staging");
    },
  });

  const modalItems = useMemo(() => {
    if (!activeCard) return [];

    if (activeCard.hasVariants) {
      const staged = Array.isArray(activeCard?.staging?.meta?.staged)
        ? activeCard.staging.meta.staged
        : [];

      return staged.map((it, idx) => {
        const variants =
          it?.variants && typeof it.variants === "object" ? it.variants : null;
        const thumb = variants
          ? pickVariantUrl(variants, { isNight: false, isLightOn: false })
          : "";

        return {
          key: `${activeCard.key}:${it?.photoId || idx}`,
          thumbnailUrl: thumb,
          alt: `Staging photo ${idx + 1}`,
          variants,
        };
      });
    }

    const images = Array.isArray(activeCard?.staging?.images)
      ? activeCard.staging.images
      : [];

    return images
      .map((img, idx) => {
        const url =
          typeof img?.storage_path === "string" ? img.storage_path.trim() : "";
        if (!url) return null;
        return {
          key: `${activeCard.key}:img:${img?.id || idx}`,
          url,
          thumbnailUrl: url,
          alt: `Staging image ${idx + 1}`,
        };
      })
      .filter(Boolean);
  }, [activeCard]);

  const getActiveUrl = useCallback(
    (item) => {
      if (!item) return "";
      if (activeCard?.hasVariants && item?.variants) {
        const effectiveIsLightOn = isVacantActive ? false : isLightOn;
        const picked = pickVariantUrl(item.variants, {
          isNight,
          isLightOn: effectiveIsLightOn,
        });
        return picked || item.thumbnailUrl || "";
      }
      return item.url || item.thumbnailUrl || "";
    },
    [activeCard?.hasVariants, isLightOn, isNight, isVacantActive],
  );

  const renderOverlay = useCallback(() => {
    if (!activeCard?.hasVariants) return null;

    const toggleWrapClass =
      "pointer-events-auto flex items-center rounded-full border border-black/10 dark:border-white/[0.15] bg-white/70 dark:bg-black/[0.45] backdrop-blur px-1 py-1 shadow-[0_14px_50px_rgba(0,0,0,0.18)] dark:shadow-[0_14px_50px_rgba(0,0,0,0.40)]";

    const toggleBtnBase =
      "w-9 h-9 rounded-full flex items-center justify-center transition-colors";

    const activeBtnClass =
      "bg-black/[0.08] dark:bg-white/[0.18] text-gray-900 dark:text-gray-50";

    const inactiveBtnClass =
      "text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-50";

    const disabledBtnClass = "opacity-40 cursor-not-allowed";

    return (
      <>
        <div className="absolute top-3 left-3">
          <div className={toggleWrapClass}>
            <button
              type="button"
              onClick={() => {
                if (isVacantActive) return;
                setIsLightOn(true);
              }}
              disabled={isVacantActive}
              className={`${toggleBtnBase} ${isLightOn ? activeBtnClass : inactiveBtnClass} ${isVacantActive ? disabledBtnClass : ""}`}
              aria-label="Lights on"
              title={
                isVacantActive
                  ? "VACANT: indoor lights are disabled"
                  : "Lights on"
              }
            >
              <Flashlight size={18} />
            </button>
            <button
              type="button"
              onClick={() => {
                if (isVacantActive) return;
                setIsLightOn(false);
              }}
              disabled={isVacantActive}
              className={`${toggleBtnBase} ${!isLightOn ? activeBtnClass : inactiveBtnClass} ${isVacantActive ? disabledBtnClass : ""}`}
              aria-label="Lights off"
              title={
                isVacantActive
                  ? "VACANT: indoor lights are disabled"
                  : "Lights off"
              }
            >
              <FlashlightOff size={18} />
            </button>
          </div>
        </div>

        <div className="absolute top-3 right-3">
          <div className={toggleWrapClass}>
            <button
              type="button"
              onClick={() => setIsNight(false)}
              className={`${toggleBtnBase} ${!isNight ? activeBtnClass : inactiveBtnClass}`}
              aria-label="Day"
              title="Day"
            >
              <Sun size={18} />
            </button>
            <button
              type="button"
              onClick={() => setIsNight(true)}
              className={`${toggleBtnBase} ${isNight ? activeBtnClass : inactiveBtnClass}`}
              aria-label="Night"
              title="Night"
            >
              <Moon size={18} />
            </button>
          </div>
        </div>
      </>
    );
  }, [activeCard?.hasVariants, isLightOn, isNight, isVacantActive]);

  const onDeleteActive = useCallback(async () => {
    setDeleteError(null);

    const stagingId = activeCard?.staging?.id
      ? String(activeCard.staging.id)
      : "";
    const title = activeCard?.title ? String(activeCard.title) : "this staging";

    if (!stagingId) {
      setDeleteError("Missing staging id");
      return;
    }

    const confirmed = window.confirm(`Delete ${title}? This cannot be undone.`);
    if (!confirmed) return;

    await deleteStagingMutation.mutateAsync({ stagingId });
  }, [activeCard?.staging?.id, activeCard?.title, deleteStagingMutation]);

  const deleteBtnLabel = deleteStagingMutation.isPending
    ? "Deleting..."
    : "Delete";

  const deleteBtnClass = deleteStagingMutation.isPending
    ? "inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-red-600 text-white text-xs font-medium opacity-70 cursor-not-allowed font-jetbrains-mono"
    : "inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-red-600 hover:bg-red-700 text-white text-xs font-medium font-jetbrains-mono";

  const headerActions = activeCard ? (
    <button
      type="button"
      onClick={onDeleteActive}
      disabled={deleteStagingMutation.isPending}
      className={deleteBtnClass}
      aria-label="Delete staging"
      title="Delete staging"
    >
      <Trash2 size={16} />
      {deleteBtnLabel}
    </button>
  ) : null;

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          Stagings
        </h3>

        <button type="button" onClick={onAddNew} className={addNewButtonClass}>
          + Add New
        </button>
      </div>

      {cards.length > 0 ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {cards.map((c) => (
            <StagingCard
              key={c.key}
              title={c.title}
              thumbUrl={c.thumbUrl}
              photoCount={c.photoCount}
              onOpen={() => setActiveCard(c)}
            />
          ))}
        </div>
      ) : (
        <div className="text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
          No stagings yet.
        </div>
      )}

      <ImageCarouselModal
        open={!!activeCard}
        title={activeCard?.title || "Staging"}
        onClose={() => setActiveCard(null)}
        items={modalItems}
        initialIndex={activeIndex}
        showThumbnails
        getActiveUrl={getActiveUrl}
        renderOverlay={renderOverlay}
        headerActions={headerActions}
      />

      {deleteError ? (
        <div className="mt-3 rounded-lg bg-red-50 dark:bg-red-900/30 p-3 text-sm text-red-700 dark:text-red-300 font-jetbrains-mono">
          {deleteError}
        </div>
      ) : null}
    </div>
  );
}
