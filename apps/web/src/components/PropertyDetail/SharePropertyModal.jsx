import { useMemo, useState, useEffect } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Copy, Link as LinkIcon, Loader2 } from "lucide-react";
import { ModalShell } from "./ModalShell";
import { titleCase } from "@/utils/formatters";

function copyToClipboard(text) {
  if (typeof navigator === "undefined" || !navigator.clipboard) {
    return false;
  }
  navigator.clipboard.writeText(text);
  return true;
}

function buildOrigin() {
  return typeof window !== "undefined" ? window.location.origin : "";
}

function buildTourSlotKey(tour) {
  const sourceType =
    tour?.source_type === "staging" || tour?.source_type === "original"
      ? tour.source_type
      : null;

  if (sourceType === "original") return "original";
  if (sourceType === "staging") {
    const st =
      typeof tour?.staging_type === "string" ? tour.staging_type.trim() : "";
    return st ? `staging:${st}` : "";
  }
  return "";
}

function slotKeyToPayload(key) {
  if (key === "original") return { sourceType: "original", stagingType: null };
  if (typeof key === "string" && key.startsWith("staging:")) {
    const stagingType = key.slice("staging:".length).trim();
    return { sourceType: "staging", stagingType: stagingType || null };
  }
  return null;
}

export function SharePropertyModal({
  open,
  onClose,
  property,
  initialClientId,
}) {
  const propertyId = property?.id;

  const { data: allClients, isLoading: clientsLoading } = useQuery({
    queryKey: ["clients", "all"],
    queryFn: async () => {
      const res = await fetch("/api/clients?type=all");
      if (!res.ok) {
        throw new Error("Failed to load customers");
      }
      return res.json();
    },
    enabled: open,
  });

  const clients = useMemo(() => {
    const list = Array.isArray(allClients) ? allClients : [];
    return list
      .filter((c) => !!c?.id)
      .map((c) => ({ id: c.id, label: c.full_name || "Customer" }));
  }, [allClients]);

  const stagings = Array.isArray(property?.stagings) ? property.stagings : [];
  const toursRaw = Array.isArray(property?.tours) ? property.tours : [];
  const contracts = Array.isArray(property?.contracts)
    ? property.contracts
    : [];

  // Only allow selecting saved tours, and dedupe by source slot.
  const tourOptions = useMemo(() => {
    const sorted = toursRaw
      .slice()
      .sort(
        (a, b) =>
          new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
      );

    const byKey = new Map();
    for (const t of sorted) {
      const key = buildTourSlotKey(t);
      if (!key) continue;
      if (!byKey.has(key)) {
        byKey.set(key, t);
      }
    }

    const keys = Array.from(byKey.keys());
    const hasOriginal = keys.includes("original");

    const stagingKeys = keys
      .filter((k) => k !== "original")
      .sort((a, b) => {
        const as = a.replace("staging:", "");
        const bs = b.replace("staging:", "");
        return as.localeCompare(bs);
      });

    const ordered = [...(hasOriginal ? ["original"] : []), ...stagingKeys];

    return ordered.map((key) => {
      if (key === "original") {
        return {
          key,
          label: "Original Virtual Tour",
        };
      }

      const st = key.replace("staging:", "");
      const name = titleCase(st || "Staging");
      return {
        key,
        label: `${name} Virtual Tour`,
      };
    });
  }, [toursRaw]);

  const [clientId, setClientId] = useState("");
  const [stagingIds, setStagingIds] = useState([]);
  const [virtualTourSlotKeys, setVirtualTourSlotKeys] = useState([]);
  const [contractIds, setContractIds] = useState([]);

  const [successUrl, setSuccessUrl] = useState(null);
  const [error, setError] = useState(null);

  // When the modal opens, reset and apply any provided client context.
  useEffect(() => {
    if (!open) {
      return;
    }
    setStagingIds([]);
    setVirtualTourSlotKeys([]);
    setContractIds([]);
    setError(null);
    setSuccessUrl(null);

    const shouldPreselect =
      typeof initialClientId === "string" && initialClientId.trim();
    if (shouldPreselect) {
      setClientId(initialClientId.trim());
    } else {
      setClientId("");
    }
  }, [open, initialClientId]);

  const createMutation = useMutation({
    mutationFn: async () => {
      setError(null);
      setSuccessUrl(null);

      if (!propertyId || !clientId) {
        throw new Error("Please select a customer.");
      }

      const virtualTourSlots = virtualTourSlotKeys
        .map((k) => slotKeyToPayload(k))
        .filter(Boolean);

      const payload = {
        propertyId,
        customerId: clientId,
        stagingIds,
        virtualTourSlots,
        contractIds,
      };

      const res = await fetch("/api/property-share-links", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not create the share link");
      }
      return res.json();
    },
    onSuccess: async (data) => {
      const origin = buildOrigin();
      const rawUrl = data?.url ? String(data.url) : "";
      const isAbsolute =
        rawUrl.startsWith("http://") || rawUrl.startsWith("https://");
      const url = rawUrl
        ? isAbsolute
          ? rawUrl
          : `${origin}${rawUrl.startsWith("/") ? "" : "/"}${rawUrl}`
        : `${origin}/share/${data.slug}`;
      setSuccessUrl(url);
      copyToClipboard(url);
    },
    onError: (err) => {
      console.error(err);
      setError(err?.message || "Could not create the share link");
    },
  });

  if (!open) return null;

  return (
    <ModalShell title="Create Share Link" onClose={onClose}>
      <div className="space-y-4 font-jetbrains-mono">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
            Customer (required)
          </label>

          <select
            value={clientId}
            onChange={(e) => setClientId(e.target.value)}
            className="mt-2 w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)]"
          >
            <option value="">
              {clientsLoading ? "Loading customers..." : "Select customer..."}
            </option>
            {clients.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}
              </option>
            ))}
          </select>

          {!clientId ? (
            <div className="mt-2 text-xs text-amber-700 dark:text-amber-300">
              Please pick exactly one customer.
            </div>
          ) : null}

          {clients.length === 0 ? (
            <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
              No customers found.
            </div>
          ) : null}
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Content
            </label>

            <div className="mt-2 rounded-lg border border-gray-200 dark:border-gray-700 p-3 space-y-3">
              <label className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-200">
                <input type="checkbox" checked readOnly disabled />
                <span>Photos (included)</span>
              </label>

              <div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  Stagings
                </div>
                <div className="mt-2 space-y-1">
                  {stagings.length === 0 ? (
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      No stagings.
                    </div>
                  ) : (
                    stagings.map((s) => {
                      const checked = stagingIds.includes(s.id);
                      const label = `${String(s.staging_type || "staging").replace(/_/g, " ")} (v${Number(s.version || 1)})`;
                      return (
                        <label
                          key={s.id}
                          className="flex items-center gap-2 py-1 text-sm text-gray-700 dark:text-gray-200"
                        >
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={() => {
                              setStagingIds((prev) => {
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
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  Virtual Tours
                </div>
                <div className="mt-2 space-y-1">
                  {tourOptions.length === 0 ? (
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      No virtual tours.
                    </div>
                  ) : (
                    tourOptions.map((opt) => {
                      const checked = virtualTourSlotKeys.includes(opt.key);
                      return (
                        <label
                          key={opt.key}
                          className="flex items-center gap-2 py-1 text-sm text-gray-700 dark:text-gray-200"
                        >
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={() => {
                              setVirtualTourSlotKeys((prev) => {
                                if (prev.includes(opt.key)) {
                                  return prev.filter((x) => x !== opt.key);
                                }
                                return [...prev, opt.key];
                              });
                            }}
                          />
                          <span>{opt.label}</span>
                        </label>
                      );
                    })
                  )}
                </div>
              </div>

              <div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  Contracts
                </div>
                <div className="mt-2 space-y-1">
                  {contracts.length === 0 ? (
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      No contracts.
                    </div>
                  ) : (
                    contracts.map((c) => {
                      const checked = contractIds.includes(c.id);
                      const meta =
                        c?.metadata && typeof c.metadata === "object"
                          ? c.metadata
                          : null;
                      const displayName = meta?.display_name
                        ? String(meta.display_name)
                        : "";
                      const labelRaw =
                        displayName || String(c.template_type || "Contract");
                      const label = labelRaw.replace(/_/g, " ");
                      return (
                        <label
                          key={c.id}
                          className="flex items-center gap-2 py-1 text-sm text-gray-700 dark:text-gray-200"
                        >
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={() => {
                              setContractIds((prev) => {
                                if (prev.includes(c.id))
                                  return prev.filter((x) => x !== c.id);
                                return [...prev, c.id];
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
            </div>
          </div>
        </div>

        {error ? (
          <div className="rounded-lg bg-red-50 dark:bg-red-900/30 p-3 text-sm text-red-600 dark:text-red-400">
            {error}
          </div>
        ) : null}

        {successUrl ? (
          <div className="rounded-lg bg-green-50 dark:bg-green-900/30 p-3 text-sm text-green-700 dark:text-green-300">
            <div className="flex items-center justify-between gap-3">
              <span className="truncate">{successUrl}</span>
              <button
                type="button"
                onClick={() => copyToClipboard(successUrl)}
                className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-white/60 dark:bg-white/10 hover:bg-white/80 dark:hover:bg-white/20"
              >
                <Copy size={16} /> Copy link
              </button>
            </div>
          </div>
        ) : null}

        <div className="flex flex-col sm:flex-row gap-3">
          <button
            type="button"
            onClick={() => createMutation.mutate()}
            disabled={!propertyId || !clientId || createMutation.isPending}
            className="inline-flex items-center justify-center gap-2 w-full px-6 py-3 bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 rounded-lg font-medium transition-colors hover:bg-gray-800 dark:hover:bg-gray-200 disabled:opacity-50"
          >
            {createMutation.isPending ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              <LinkIcon size={18} />
            )}
            Generate
          </button>

          <button
            type="button"
            onClick={onClose}
            className="inline-flex items-center justify-center gap-2 w-full px-6 py-3 rounded-lg font-medium transition-colors border border-gray-200 dark:border-gray-700 bg-white dark:bg-[#262626] text-gray-900 dark:text-gray-100 hover:bg-gray-50 dark:hover:bg-gray-800"
          >
            Cancel
          </button>
        </div>
      </div>
    </ModalShell>
  );
}
