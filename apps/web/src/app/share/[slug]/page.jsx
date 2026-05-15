import { useState, useMemo, useCallback, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ChevronLeft,
  ChevronRight,
  Download,
  Moon,
  Sun,
  Flashlight,
  FlashlightOff,
  Eye,
  Images,
} from "lucide-react";
import { StatusBanner } from "@/components/StatusBanner";
import { formatIntegerForInput, titleCase } from "@/utils/formatters";
import ImageCarouselModal from "@/components/ImageCarouselModal";

function buildViewerUrl({ dataUrl, returnUrl }) {
  const data = encodeURIComponent(dataUrl);
  const ret = returnUrl ? `&returnUrl=${encodeURIComponent(returnUrl)}` : "";
  return `/viewer/index.html?data=${data}&mode=readonly${ret}`;
}

function OptionalHeader({ agent }) {
  const companyName = agent?.company_name ? String(agent.company_name) : "";
  const logoUrl = agent?.company_logo_url ? String(agent.company_logo_url) : "";

  if (!companyName && !logoUrl) return null;

  return (
    <div className="mb-6 flex items-center gap-3 border-b border-gray-200 pb-4">
      {logoUrl ? (
        <img
          src={logoUrl}
          alt={companyName || "Company"}
          className="h-10 w-10 rounded-md object-cover border border-gray-200"
        />
      ) : null}
      {companyName ? (
        <div className="text-lg font-semibold text-gray-900 font-jetbrains-mono">
          {companyName}
        </div>
      ) : null}
    </div>
  );
}

function OptionalFooter({ agent }) {
  const companyName = agent?.company_name ? String(agent.company_name) : "";
  const agentName = agent?.agent_name ? String(agent.agent_name) : "";
  const email = agent?.agent_email ? String(agent.agent_email) : "";
  const phone = agent?.agent_phone ? String(agent.agent_phone) : "";

  const hasAny = !!(companyName || agentName || email || phone);
  if (!hasAny) return null;

  return (
    <div className="mt-10 border-t border-gray-200 pt-6">
      <div className="text-sm text-gray-900 font-jetbrains-mono">
        {companyName ? <div>{companyName}</div> : null}
        {agentName ? <div>{agentName}</div> : null}
        {phone ? <div>{phone}</div> : null}
        {email ? <div>{email}</div> : null}
      </div>
    </div>
  );
}

function PhotoGallery({ photoUrls, onOpen }) {
  const [index, setIndex] = useState(0);

  const safeIndex = Math.min(
    Math.max(index, 0),
    Math.max(0, photoUrls.length - 1),
  );
  const activeUrl = photoUrls.length > 0 ? photoUrls[safeIndex] : null;

  const onPrev = () => {
    setIndex((i) => (i <= 0 ? photoUrls.length - 1 : i - 1));
  };

  const onNext = () => {
    setIndex((i) => (i >= photoUrls.length - 1 ? 0 : i + 1));
  };

  if (!activeUrl) {
    return (
      <div className="rounded-xl border border-dashed border-gray-300 p-8 text-center text-sm text-gray-600 font-jetbrains-mono">
        No photos.
      </div>
    );
  }

  return (
    <div>
      <div className="relative overflow-hidden rounded-xl border border-gray-200 bg-white">
        <button
          type="button"
          onClick={() => onOpen?.(safeIndex)}
          className="block w-full"
          aria-label="Open photos"
          title="Open"
        >
          <img
            src={activeUrl}
            alt="Property photo"
            className="w-full h-[320px] sm:h-[420px] object-cover"
          />
        </button>

        {photoUrls.length > 1 ? (
          <div className="absolute inset-0 flex items-center justify-between px-3 pointer-events-none">
            <button
              type="button"
              onClick={onPrev}
              className="pointer-events-auto inline-flex items-center justify-center h-10 w-10 rounded-full bg-white/80 hover:bg-white border border-gray-200"
              aria-label="Previous"
            >
              <ChevronLeft size={18} />
            </button>
            <button
              type="button"
              onClick={onNext}
              className="pointer-events-auto inline-flex items-center justify-center h-10 w-10 rounded-full bg-white/80 hover:bg-white border border-gray-200"
              aria-label="Next"
            >
              <ChevronRight size={18} />
            </button>
          </div>
        ) : null}
      </div>

      {photoUrls.length > 1 ? (
        <div className="mt-3 flex gap-2 overflow-x-auto pb-1">
          {photoUrls.map((u, i) => {
            const isActive = i === safeIndex;
            const ring = isActive ? "ring-2 ring-[var(--brand)]" : "";
            return (
              <button
                type="button"
                key={`${u}-${i}`}
                onClick={() => setIndex(i)}
                className={`shrink-0 rounded-lg overflow-hidden border border-gray-200 ${ring}`}
                aria-label={`Photo ${i + 1}`}
              >
                <img
                  src={u}
                  alt="Thumbnail"
                  className="h-16 w-24 object-cover"
                />
              </button>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

function PropertyDetails({ property }) {
  const addressParts = [];
  if (property?.address_line) addressParts.push(property.address_line);
  if (property?.city) addressParts.push(property.city);
  if (property?.country) addressParts.push(property.country);

  const address = addressParts.join(", ");

  const priceValue =
    property?.price !== null && property?.price !== undefined
      ? `${formatIntegerForInput(property.price)} ${property?.currency || ""}`.trim()
      : "—";

  const rows = [
    { label: "Address", value: address || "—" },
    { label: "Price", value: priceValue },
    {
      label: "Housing type",
      value: property?.housing_type ? titleCase(property.housing_type) : "—",
    },
    {
      label: "Layout",
      value: property?.housing_shape ? String(property.housing_shape) : "—",
    },
    { label: "Bedrooms", value: property?.bedrooms ?? "—" },
    { label: "Living rooms", value: property?.living_rooms ?? "—" },
    { label: "Bathrooms", value: property?.bathrooms ?? "—" },
    {
      label: "Gross area (sqm)",
      value: property?.gross_area_sqm ?? property?.size_sqm ?? "—",
    },
    { label: "Net area (sqm)", value: property?.net_area_sqm ?? "—" },
    { label: "Floor", value: property?.floor_number ?? "—" },
    { label: "Total floors", value: property?.total_floors ?? "—" },
  ];

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-5">
      <div className="text-sm font-semibold text-gray-900 mb-4 font-jetbrains-mono">
        Property details
      </div>
      <div className="space-y-3">
        {rows.map((r) => (
          <div key={r.label} className="flex items-start justify-between gap-4">
            <div className="text-xs text-gray-600 font-jetbrains-mono">
              {r.label}
            </div>
            <div className="text-sm text-gray-900 font-jetbrains-mono text-right">
              {r.value === null || r.value === undefined || r.value === ""
                ? "—"
                : String(r.value)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function pickFirstNonEmptyUrl(values) {
  for (const v of values) {
    if (typeof v === "string" && v.trim().length > 0) return v.trim();
  }
  return "";
}

function getVariantUrlFromVariants(variants, { isNight, isLightOn }) {
  if (!variants || typeof variants !== "object") return "";

  const desiredKey = `${isNight ? "night" : "day"}_light_${isLightOn ? "on" : "off"}`;
  const otherTimeSameLightKey = `${isNight ? "day" : "night"}_light_${isLightOn ? "on" : "off"}`;
  const sameTimeOtherLightKey = `${isNight ? "night" : "day"}_light_${isLightOn ? "off" : "on"}`;
  const otherTimeOtherLightKey = `${isNight ? "day" : "night"}_light_${isLightOn ? "off" : "on"}`;

  const candidates = [
    variants?.[desiredKey],
    variants?.[otherTimeSameLightKey],
    variants?.[sameTimeOtherLightKey],
    variants?.[otherTimeOtherLightKey],
  ];

  for (const v of candidates) {
    if (typeof v === "string" && v.trim()) return v.trim();
  }

  return "";
}

function ShareStagingThumb({ title, thumbUrl, photoCount, onOpen }) {
  return (
    <button
      type="button"
      onClick={onOpen}
      className="rounded-lg overflow-hidden border border-gray-200 bg-white text-left hover:bg-gray-50"
      aria-label="Open staging"
      title="Open"
    >
      <div className="relative">
        {thumbUrl ? (
          <img
            src={thumbUrl}
            alt="Staging"
            className="w-full h-40 sm:h-44 object-cover"
            loading="lazy"
          />
        ) : (
          <div className="w-full h-40 sm:h-44 flex items-center justify-center bg-gray-50 text-gray-600 font-jetbrains-mono text-sm">
            No staging image.
          </div>
        )}

        {Number(photoCount || 0) > 1 ? (
          <div className="absolute bottom-2 right-2 inline-flex items-center gap-1 rounded-full bg-white/90 border border-gray-200 px-2 py-1 text-xs text-gray-900 font-jetbrains-mono">
            <Images size={14} />
            <span>{Number(photoCount || 0)}</span>
          </div>
        ) : null}
      </div>

      {title ? (
        <div className="px-4 py-3 border-t border-gray-200">
          <div className="text-xs text-gray-700 font-jetbrains-mono">
            {title}
          </div>
        </div>
      ) : null}
    </button>
  );
}

export default function SharePage(props) {
  const slug = props?.params?.slug;

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["share", slug],
    queryFn: async () => {
      const res = await fetch(`/api/share/${slug}`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        const err = new Error(body?.error || "Not found");
        err.status = res.status;
        throw err;
      }
      return res.json();
    },
    enabled: !!slug,
    refetchOnWindowFocus: false,
    retry: (failureCount, err) => {
      // Don't retry on 404 (not found) or 410 (expired) — these are definitive.
      const status = Number(err?.status || 0);
      if (status === 404 || status === 410) return false;
      // Retry up to 2 times for transient errors (500, network, etc.)
      return failureCount < 2;
    },
  });

  const property = data?.property || null;
  const customer = data?.customer || null;
  const agent = data?.agent || null;

  const photoUrls = Array.isArray(property?.photo_download_urls)
    ? property.photo_download_urls
    : [];

  const [photosOpen, setPhotosOpen] = useState(false);
  const [photosIndex, setPhotosIndex] = useState(0);

  const photoItems = useMemo(() => {
    return photoUrls
      .filter((u) => typeof u === "string" && u.trim().length > 0)
      .map((u, i) => ({
        key: `${u}-${i}`,
        url: u,
        thumbnailUrl: u,
        alt: "Property photo",
      }));
  }, [photoUrls]);

  const stagings = Array.isArray(property?.stagings) ? property.stagings : [];

  const stagingGroups = useMemo(() => {
    const out = [];

    for (const s of stagings) {
      const stagingType =
        typeof s?.staging_type === "string" ? String(s.staging_type) : "";

      const title = stagingType ? titleCase(stagingType) : "Staging";

      const stagedItems = Array.isArray(s?.staged_items) ? s.staged_items : [];

      if (stagedItems.length > 0) {
        const firstThumb = getVariantUrlFromVariants(
          stagedItems?.[0]?.variants,
          {
            isNight: false,
            isLightOn: false,
          },
        );

        out.push({
          key: String(s.id),
          title,
          stagingType,
          hasVariants: true,
          thumbUrl: firstThumb,
          items: stagedItems.map((it, idx) => {
            const t = getVariantUrlFromVariants(it?.variants, {
              isNight: false,
              isLightOn: false,
            });
            return {
              key: it?.key || `${s.id}:${idx}`,
              thumbnailUrl: t,
              variants: it?.variants || null,
              alt: `Staging photo ${idx + 1}`,
            };
          }),
        });
        continue;
      }

      // Backward compat
      const images = Array.isArray(s?.images) ? s.images : [];
      const urls = images
        .map((i) => i?.download_url)
        .filter((u) => typeof u === "string" && u.trim().length > 0)
        .map((u) => u.trim());

      out.push({
        key: String(s.id),
        title,
        stagingType,
        hasVariants: false,
        thumbUrl: urls[0] || "",
        items: urls.map((u, idx) => ({
          key: `${s.id}:img:${idx}`,
          url: u,
          thumbnailUrl: u,
          alt: `Staging image ${idx + 1}`,
        })),
      });
    }

    return out;
  }, [stagings]);

  const [activeStaging, setActiveStaging] = useState(null);
  const [stagingIndex, setStagingIndex] = useState(0);
  const [isNight, setIsNight] = useState(false);
  const [isLightOn, setIsLightOn] = useState(false);

  const isVacantActive = activeStaging?.stagingType === "vacant";

  useEffect(() => {
    if (!activeStaging) return;
    setStagingIndex(0);
    setIsNight(false);
    setIsLightOn(false);
  }, [activeStaging]);

  const getActiveStagingUrl = useCallback(
    (item) => {
      if (!item) return "";
      if (activeStaging?.hasVariants && item?.variants) {
        const effectiveIsLightOn = isVacantActive ? false : isLightOn;
        const u = getVariantUrlFromVariants(item.variants, {
          isNight,
          isLightOn: effectiveIsLightOn,
        });
        return u || item.thumbnailUrl || "";
      }
      return item.url || item.thumbnailUrl || "";
    },
    [activeStaging?.hasVariants, isLightOn, isNight, isVacantActive],
  );

  const renderStagingOverlay = useCallback(() => {
    if (!activeStaging?.hasVariants) return null;

    const toggleWrapClass =
      "pointer-events-auto flex items-center rounded-full border border-black/10 bg-white/70 backdrop-blur px-1 py-1 shadow-[0_14px_50px_rgba(0,0,0,0.18)]";

    const toggleBtnBase =
      "w-9 h-9 rounded-full flex items-center justify-center transition-colors";

    const activeBtnClass = "bg-black/[0.08] text-gray-900";

    const inactiveBtnClass = "text-gray-700 hover:text-gray-900";

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
  }, [activeStaging?.hasVariants, isLightOn, isNight, isVacantActive]);

  const tours = Array.isArray(property?.tours) ? property.tours : [];
  const contracts = Array.isArray(property?.contracts)
    ? property.contracts
    : [];

  const customerName = customer?.full_name ? String(customer.full_name) : "";

  const returnUrl = slug ? `/share/${encodeURIComponent(slug)}` : "/";

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-6xl mx-auto">
          <p className="text-gray-600 font-jetbrains-mono">Loading...</p>
        </div>
      </div>
    );
  }

  if (error) {
    const status = Number(error?.status || 0);
    const message = String(error?.message || "");

    const isExpired =
      status === 410 || message.toLowerCase().includes("expired");

    const isTransient = status >= 500 || status === 0;

    const title = isExpired ? "Link expired" : "Not available";
    const text = isExpired
      ? "This share link has expired."
      : "This item is no longer available.";

    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-6xl mx-auto">
          <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
            <StatusBanner variant="info" title={title}>
              {text}
            </StatusBanner>
            {isTransient ? (
              <div className="mt-4 text-center">
                <button
                  type="button"
                  onClick={() => refetch()}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-900 text-white text-sm font-medium hover:bg-gray-800 font-jetbrains-mono"
                >
                  Try again
                </button>
              </div>
            ) : null}
          </div>
        </div>
      </div>
    );
  }

  if (!property) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-6xl mx-auto">
          <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
            <StatusBanner variant="info" title="Not available">
              This item is no longer available.
            </StatusBanner>
          </div>
        </div>
      </div>
    );
  }

  const fixedMessage = customerName
    ? `Hello ${customerName},\nBelow you can find all the details of the property shared with you.`
    : `Hello,\nBelow you can find all the details of the property shared with you.`;

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-6xl mx-auto px-4 sm:px-8 py-8 sm:py-12">
        <OptionalHeader agent={agent} />

        <div className="rounded-xl border border-gray-200 bg-white p-5">
          <div className="whitespace-pre-line text-sm text-gray-900 font-jetbrains-mono">
            {fixedMessage}
          </div>
        </div>

        <div className="mt-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-7">
            <PhotoGallery
              photoUrls={photoUrls}
              onOpen={(idx) => {
                setPhotosIndex(idx);
                setPhotosOpen(true);
              }}
            />
          </div>
          <div className="lg:col-span-5">
            <PropertyDetails property={property} />
          </div>
        </div>

        {stagingGroups.length > 0 ? (
          <div className="mt-8 rounded-xl border border-gray-200 bg-white p-5">
            <div className="text-sm font-semibold text-gray-900 mb-4 font-jetbrains-mono">
              Stagings
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {stagingGroups.map((s) => (
                <ShareStagingThumb
                  key={s.key}
                  title={s.title}
                  thumbUrl={s.thumbUrl}
                  photoCount={Array.isArray(s.items) ? s.items.length : 0}
                  onOpen={() => setActiveStaging(s)}
                />
              ))}
            </div>
          </div>
        ) : null}

        {/* 4) Selected virtual tour links */}
        {tours.length > 0 ? (
          <div className="mt-8 rounded-xl border border-gray-200 bg-white p-5">
            <div className="text-sm font-semibold text-gray-900 mb-4 font-jetbrains-mono">
              Virtual tours
            </div>
            <div className="space-y-2">
              {tours.map((t) => {
                const isStaging = t?.source_type === "staging";
                const stagingType =
                  typeof t?.staging_type === "string" ? t.staging_type : "";

                const label = isStaging
                  ? `${titleCase(stagingType || "Staging")} Virtual Tour`
                  : "Original Virtual Tour";

                const payload =
                  t?.tour_payload && typeof t.tour_payload === "object"
                    ? t.tour_payload
                    : {};
                const dataUrl = payload?.data_url
                  ? String(payload.data_url)
                  : "";

                const href = dataUrl
                  ? buildViewerUrl({ dataUrl, returnUrl })
                  : `/share/${encodeURIComponent(slug)}/tours/${encodeURIComponent(String(t.id))}`;

                return (
                  <a
                    key={t.id}
                    href={href}
                    target="_blank"
                    rel="noreferrer"
                    className="flex items-center justify-between gap-3 rounded-lg border border-gray-200 px-4 py-3 hover:bg-gray-50"
                  >
                    <div className="text-sm text-gray-900 font-jetbrains-mono truncate">
                      {label}
                    </div>
                    <div className="text-xs text-gray-600 font-jetbrains-mono">
                      View
                    </div>
                  </a>
                );
              })}
            </div>
          </div>
        ) : null}

        {/* 5) Selected contracts */}
        {contracts.length > 0 ? (
          <div className="mt-8 rounded-xl border border-gray-200 bg-white p-5">
            <div className="text-sm font-semibold text-gray-900 mb-4 font-jetbrains-mono">
              Contracts
            </div>
            <div className="space-y-2">
              {contracts.map((c) => {
                const downloadUrl = c.pdf_download_url || null;
                const viewUrl = downloadUrl
                  ? `${downloadUrl}?disposition=inline`
                  : null;
                const hasPdf = !!c.has_pdf;

                const meta =
                  c?.metadata && typeof c.metadata === "object"
                    ? c.metadata
                    : null;
                const displayName = meta?.display_name
                  ? String(meta.display_name)
                  : null;

                const label = displayName
                  ? displayName
                  : titleCase(c.template_type);

                return (
                  <div
                    key={c.id}
                    className="rounded-lg border border-gray-200 px-4 py-3"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="min-w-0">
                        <div className="text-sm text-gray-900 font-jetbrains-mono truncate">
                          {label}
                        </div>
                      </div>

                      {hasPdf && downloadUrl ? (
                        <div className="flex items-center gap-2">
                          {viewUrl ? (
                            <a
                              href={viewUrl}
                              target="_blank"
                              rel="noreferrer"
                              className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-gray-200 bg-white text-gray-900 text-xs font-medium hover:bg-gray-50 font-jetbrains-mono"
                            >
                              <Eye size={16} />
                              View
                            </a>
                          ) : null}
                          <a
                            href={downloadUrl}
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-900 text-white text-xs font-medium hover:bg-gray-800 font-jetbrains-mono"
                          >
                            <Download size={16} />
                            Download
                          </a>
                        </div>
                      ) : null}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ) : null}

        <OptionalFooter agent={agent} />
      </div>

      <ImageCarouselModal
        open={photosOpen}
        title="Photos"
        onClose={() => setPhotosOpen(false)}
        items={photoItems}
        initialIndex={photosIndex}
        showThumbnails
      />

      <ImageCarouselModal
        open={!!activeStaging}
        title={activeStaging?.title || "Staging"}
        onClose={() => setActiveStaging(null)}
        items={activeStaging?.items || []}
        initialIndex={stagingIndex}
        showThumbnails
        getActiveUrl={getActiveStagingUrl}
        renderOverlay={renderStagingOverlay}
      />
    </div>
  );
}
