import { useMemo, useState, useCallback } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import useUser from "@/utils/useUser";
import { Header } from "../../components/Header";
import {
  Check,
  Copy,
  Link as LinkIcon,
  Loader2,
  Shield,
  Timer,
  XCircle,
  Eye,
} from "lucide-react";
import {
  SHARE_LINK_DEFAULT_EXPIRY_DAYS,
  SHARE_LINK_MAX_EXPIRY_DAYS,
  normalizeExpiryDays,
} from "@/utils/shareLinksConfig";

function copyToClipboard(text) {
  if (typeof navigator === "undefined" || !navigator.clipboard) {
    return false;
  }
  navigator.clipboard.writeText(text);
  return true;
}

function titleCase(s) {
  if (!s) return "";
  return String(s)
    .split("_")
    .map((p) => p.charAt(0).toUpperCase() + p.slice(1))
    .join(" ");
}

function isExpired(expiresAt) {
  if (!expiresAt) return false;
  const t = new Date(expiresAt).getTime();
  if (Number.isNaN(t)) return false;
  return t < Date.now();
}

function getLastViewedAt(meta) {
  const access = meta && typeof meta === "object" ? meta.access : null;
  const arr = Array.isArray(access) ? access : [];
  let max = null;
  for (const a of arr) {
    const ts = a?.timestamp;
    if (!ts) continue;
    const d = new Date(ts);
    if (Number.isNaN(d.getTime())) continue;
    if (!max || d.getTime() > max.getTime()) max = d;
  }
  return max ? max.toISOString() : null;
}

export default function LinksPage() {
  const queryClient = useQueryClient();
  const { data: user, loading: userLoading } = useUser();

  const [propertyId, setPropertyId] = useState("");
  const [clientId, setClientId] = useState("");
  const [includeStagingIds, setIncludeStagingIds] = useState([]);
  const [includeTourId, setIncludeTourId] = useState("");
  const [includeContractIds, setIncludeContractIds] = useState([]);

  // NEW: auto-expire by default (configurable days)
  const [expiresInDays, setExpiresInDays] = useState(
    SHARE_LINK_DEFAULT_EXPIRY_DAYS,
  );

  const [successUrl, setSuccessUrl] = useState(null);
  const [error, setError] = useState(null);

  const supportSafeSuccessUrl = useMemo(() => {
    if (!successUrl) return null;
    try {
      const u = new URL(successUrl);
      u.searchParams.set("view", "readonly");
      return u.toString();
    } catch {
      return null;
    }
  }, [successUrl]);

  const { data: properties = [] } = useQuery({
    queryKey: ["properties", user?.id, "links"],
    queryFn: async () => {
      const res = await fetch("/api/properties");
      if (!res.ok) {
        throw new Error("Failed to load properties");
      }
      return res.json();
    },
    enabled: !!user?.id,
  });

  const { data: clients = [] } = useQuery({
    queryKey: ["clients", user?.id, "links"],
    queryFn: async () => {
      const res = await fetch("/api/clients");
      if (!res.ok) {
        throw new Error("Failed to load clients");
      }
      return res.json();
    },
    enabled: !!user?.id,
  });

  const { data: propertyDetail } = useQuery({
    queryKey: ["property", user?.id, propertyId, "links"],
    queryFn: async () => {
      const res = await fetch(`/api/properties/${propertyId}`);
      if (!res.ok) {
        throw new Error("Failed to load property detail");
      }
      return res.json();
    },
    enabled: !!user?.id && !!propertyId,
  });

  const stagings = propertyDetail?.stagings || [];
  const tours = propertyDetail?.tours || [];
  const contracts = propertyDetail?.contracts || [];

  const latestVersionByType = useMemo(() => {
    const map = {};
    for (const s of stagings) {
      const t = s.staging_type;
      const v = Number(s.version || 1);
      if (!map[t] || v > map[t]) map[t] = v;
    }
    return map;
  }, [stagings]);

  const formatStagingLabel = useCallback(
    (s) => {
      const t = titleCase(s?.staging_type || "staging");
      const v = Number(s?.version || 1);
      const isLatest = latestVersionByType[s?.staging_type] === v;
      return `${t} (v${v}${isLatest ? " – latest" : ""})`;
    },
    [latestVersionByType],
  );

  const { data: shareLinks = [], refetch: refetchLinks } = useQuery({
    queryKey: ["share-links", user?.id],
    queryFn: async () => {
      const res = await fetch("/api/share-links");
      if (!res.ok) {
        throw new Error("Failed to load share links");
      }
      return res.json();
    },
    enabled: !!user?.id,
  });

  const createMutation = useMutation({
    mutationFn: async () => {
      setError(null);
      setSuccessUrl(null);

      const payload = {
        propertyId,
        clientId,
        includeStagingIds,
        includeTourId: includeTourId || null,
        includeContractIds,
        expiresInDays: normalizeExpiryDays(expiresInDays),
      };

      const res = await fetch("/api/share-links", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to create share link");
      }
      return res.json();
    },
    onSuccess: async (data) => {
      const origin =
        typeof window !== "undefined" ? window.location.origin : "";
      setSuccessUrl(`${origin}/share/${data.slug}`);
      await refetchLinks();
    },
    onError: (err) => {
      console.error(err);
      setError(err?.message || "Failed to create share link");
    },
  });

  const disableMutation = useMutation({
    mutationFn: async ({ id }) => {
      const res = await fetch(`/api/share-links/${id}/disable`, {
        method: "POST",
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to disable link");
      }
      return res.json();
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: ["share-links", user?.id],
      });
    },
    onError: (err) => {
      console.error(err);
      setError(err?.message || "Failed to disable link");
    },
  });

  const extendMutation = useMutation({
    mutationFn: async ({ id, extendDays }) => {
      const res = await fetch(`/api/share-links/${id}/extend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ extendDays }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to extend expiry");
      }
      return res.json();
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: ["share-links", user?.id],
      });
    },
    onError: (err) => {
      console.error(err);
      setError(err?.message || "Failed to extend expiry");
    },
  });

  if (userLoading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E]">
        <Header />
        <div className="pt-16 flex items-center justify-center min-h-screen">
          <p className="text-gray-600 dark:text-gray-400 font-jetbrains-mono">
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

  const selectedStagingCount = includeStagingIds.length;
  const selectedContractCount = includeContractIds.length;

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E]">
      <Header />

      <div className="pt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-8 py-8 sm:py-12">
          <div className="mb-8">
            <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-gray-100 mb-2 font-jetbrains-mono">
              Share Links
            </h1>
            <p className="text-gray-600 dark:text-gray-300 font-jetbrains-mono">
              Create read-only links for clients.
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 font-jetbrains-mono">
                Create Link
              </h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                    Property
                  </label>
                  <select
                    value={propertyId}
                    onChange={(e) => {
                      setPropertyId(e.target.value);
                      setIncludeStagingIds([]);
                      setIncludeTourId("");
                      setIncludeContractIds([]);
                    }}
                    className="mt-2 w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                  >
                    <option value="">Select property...</option>
                    {properties.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.title}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                    Client
                  </label>
                  <select
                    value={clientId}
                    onChange={(e) => setClientId(e.target.value)}
                    className="mt-2 w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                  >
                    <option value="">Select client...</option>
                    {clients.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.full_name}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                    Include Stagings ({selectedStagingCount})
                  </label>
                  <div className="mt-2 rounded-lg border border-gray-200 dark:border-gray-700 p-3 max-h-40 overflow-auto">
                    {stagings.length === 0 ? (
                      <div className="text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                        No stagings available.
                      </div>
                    ) : (
                      stagings.map((s) => {
                        const checked = includeStagingIds.includes(s.id);
                        const label = formatStagingLabel(s);
                        return (
                          <label
                            key={s.id}
                            className="flex items-center gap-2 py-1 text-sm text-gray-700 dark:text-gray-200 font-jetbrains-mono"
                          >
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => {
                                setIncludeStagingIds((prev) => {
                                  if (prev.includes(s.id))
                                    return prev.filter((x) => x !== s.id);
                                  return [...prev, s.id];
                                });
                              }}
                            />
                            <span>{label}</span>
                          </label>
                        );
                      })
                    )}
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                    Include Virtual Tour
                  </label>
                  <select
                    value={includeTourId}
                    onChange={(e) => setIncludeTourId(e.target.value)}
                    className="mt-2 w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                    disabled={!propertyId}
                  >
                    <option value="">No tour</option>
                    {tours.map((t) => (
                      <option key={t.id} value={t.id}>
                        {t.tour_type} •{" "}
                        {new Date(t.created_at).toLocaleDateString()}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                    Include Contracts ({selectedContractCount})
                  </label>
                  <div className="mt-2 rounded-lg border border-gray-200 dark:border-gray-700 p-3 max-h-40 overflow-auto">
                    {contracts.length === 0 ? (
                      <div className="text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                        No contracts available.
                      </div>
                    ) : (
                      contracts.map((c) => {
                        const checked = includeContractIds.includes(c.id);
                        return (
                          <label
                            key={c.id}
                            className="flex items-center gap-2 py-1 text-sm text-gray-700 dark:text-gray-200 font-jetbrains-mono"
                          >
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => {
                                setIncludeContractIds((prev) => {
                                  if (prev.includes(c.id))
                                    return prev.filter((x) => x !== c.id);
                                  return [...prev, c.id];
                                });
                              }}
                            />
                            <span>{c.template_type}</span>
                          </label>
                        );
                      })
                    )}
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                    Expiry (days)
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={SHARE_LINK_MAX_EXPIRY_DAYS}
                    value={expiresInDays}
                    onChange={(e) => setExpiresInDays(e.target.value)}
                    className="mt-2 w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                  />
                  <div className="mt-1 text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
                    Default is {SHARE_LINK_DEFAULT_EXPIRY_DAYS} days. Creating a
                    new link for the same property + client will disable older
                    active links.
                  </div>
                </div>

                {error ? (
                  <div className="rounded-lg bg-red-50 dark:bg-red-900/30 p-3 text-sm text-red-600 dark:text-red-400 font-jetbrains-mono">
                    {error}
                  </div>
                ) : null}

                {successUrl ? (
                  <div className="rounded-lg bg-green-50 dark:bg-green-900/30 p-3 text-sm text-green-700 dark:text-green-300 font-jetbrains-mono">
                    <div className="flex items-center justify-between gap-3">
                      <span className="truncate">{successUrl}</span>
                      <button
                        type="button"
                        onClick={() => copyToClipboard(successUrl)}
                        className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-white/60 dark:bg-white/10 hover:bg-white/80 dark:hover:bg-white/20"
                      >
                        <Copy size={16} />
                        Copy
                      </button>
                    </div>

                    {supportSafeSuccessUrl ? (
                      <div className="mt-2 flex items-center justify-between gap-3">
                        <span className="truncate text-xs text-green-800/80 dark:text-green-200/80">
                          {supportSafeSuccessUrl}
                        </span>
                        <button
                          type="button"
                          onClick={() => copyToClipboard(supportSafeSuccessUrl)}
                          className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-white/60 dark:bg-white/10 hover:bg-white/80 dark:hover:bg-white/20"
                          title="Same URL with ?view=readonly hint"
                        >
                          <Shield size={16} />
                          Copy support-safe
                        </button>
                      </div>
                    ) : null}
                  </div>
                ) : null}

                <button
                  type="button"
                  disabled={
                    !propertyId || !clientId || createMutation.isPending
                  }
                  onClick={() => createMutation.mutate()}
                  className="inline-flex items-center justify-center gap-2 w-full px-6 py-3 bg-[var(--brand)] hover:bg-[var(--brandHover)] text-white rounded-lg font-medium transition-colors disabled:opacity-50 font-jetbrains-mono"
                >
                  {createMutation.isPending ? (
                    <Loader2 size={18} className="animate-spin" />
                  ) : (
                    <LinkIcon size={18} />
                  )}
                  Create Share Link
                </button>
              </div>
            </div>

            <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 font-jetbrains-mono">
                Existing Links
              </h2>

              {shareLinks.length === 0 ? (
                <div className="text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                  No share links yet.
                </div>
              ) : (
                <div className="space-y-3">
                  {shareLinks.map((sl) => {
                    const origin =
                      typeof window !== "undefined"
                        ? window.location.origin
                        : "";
                    const url = `${origin}/share/${sl.slug}`;
                    const expired = isExpired(sl.expires_at);
                    const expiryText = sl.expires_at
                      ? new Date(sl.expires_at).toLocaleString()
                      : "No expiry";

                    const lastViewedIso = getLastViewedAt(sl.meta);
                    const lastViewedText = lastViewedIso
                      ? new Date(lastViewedIso).toLocaleString()
                      : "Never";

                    let supportSafe = null;
                    try {
                      const u = new URL(url);
                      u.searchParams.set("view", "readonly");
                      supportSafe = u.toString();
                    } catch {
                      supportSafe = null;
                    }

                    return (
                      <div
                        key={sl.id}
                        className="rounded-lg border border-gray-200 dark:border-gray-700 p-3"
                      >
                        <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                          {sl.property_title || "Property"} •{" "}
                          {sl.client_name || "Client"}
                        </div>

                        <div className="mt-1 flex items-center gap-3 text-xs font-jetbrains-mono">
                          <span
                            className={
                              expired
                                ? "inline-flex items-center gap-1 text-red-600 dark:text-red-400"
                                : "inline-flex items-center gap-1 text-green-700 dark:text-green-300"
                            }
                          >
                            {expired ? (
                              <XCircle size={14} />
                            ) : (
                              <Check size={14} />
                            )}
                            {expired ? "Expired" : "Active"}
                          </span>
                          <span className="text-gray-600 dark:text-gray-400 inline-flex items-center gap-1">
                            <Timer size={14} />
                            Expires: {expiryText}
                          </span>
                          <span className="text-gray-600 dark:text-gray-400 inline-flex items-center gap-1">
                            <Eye size={14} />
                            Last viewed: {lastViewedText}
                          </span>
                        </div>

                        <div className="mt-2 flex items-center justify-between gap-3">
                          <a
                            href={url}
                            target="_blank"
                            rel="noreferrer"
                            className="text-sm text-[var(--brandDark)] dark:text-[var(--brand)] hover:underline font-jetbrains-mono truncate"
                          >
                            {url}
                          </a>
                          <div className="flex items-center gap-2">
                            <button
                              type="button"
                              onClick={() => copyToClipboard(url)}
                              className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700"
                            >
                              <Copy size={16} />
                              <span className="text-sm font-jetbrains-mono">
                                Copy
                              </span>
                            </button>

                            {supportSafe ? (
                              <button
                                type="button"
                                onClick={() => copyToClipboard(supportSafe)}
                                className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700"
                                title="Same URL with ?view=readonly hint"
                              >
                                <Shield size={16} />
                                <span className="text-sm font-jetbrains-mono">
                                  Support-safe
                                </span>
                              </button>
                            ) : null}
                          </div>
                        </div>

                        <div className="mt-2 flex flex-wrap gap-2">
                          <button
                            type="button"
                            onClick={() =>
                              disableMutation.mutate({ id: sl.id })
                            }
                            disabled={disableMutation.isPending || expired}
                            className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-white dark:bg-[#262626] border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 disabled:opacity-50"
                            title={
                              expired
                                ? "Already expired"
                                : "Disable immediately"
                            }
                          >
                            <XCircle size={16} />
                            <span className="text-sm font-jetbrains-mono">
                              Disable
                            </span>
                          </button>

                          <button
                            type="button"
                            onClick={() =>
                              extendMutation.mutate({
                                id: sl.id,
                                extendDays: 7,
                              })
                            }
                            disabled={extendMutation.isPending}
                            className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-white dark:bg-[#262626] border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 disabled:opacity-50"
                          >
                            <Timer size={16} />
                            <span className="text-sm font-jetbrains-mono">
                              Extend 7d
                            </span>
                          </button>

                          <button
                            type="button"
                            onClick={() =>
                              extendMutation.mutate({
                                id: sl.id,
                                extendDays: 30,
                              })
                            }
                            disabled={extendMutation.isPending}
                            className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-white dark:bg-[#262626] border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 disabled:opacity-50"
                          >
                            <Timer size={16} />
                            <span className="text-sm font-jetbrains-mono">
                              Extend 30d
                            </span>
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
