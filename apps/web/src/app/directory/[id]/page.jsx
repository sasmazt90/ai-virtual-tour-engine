import { Header } from "@/components/Header";
import { useMemo, useState, useCallback, useEffect } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import useUser from "@/utils/useUser";
import {
  Copy,
  Mail,
  Phone,
  User as UserIcon,
  ArrowLeft,
  Link as LinkIcon,
  Pencil,
  Trash2,
  X,
  FileText,
} from "lucide-react";
import { useCalendarEvents } from "@/hooks/useCalendarEvents";
import { EventCard } from "@/components/Calendar/EventCard";
import { EventDetailModal } from "@/components/Calendar/EventDetailModal";
import { useCountriesAndCities } from "@/hooks/useCountriesAndCities";
import CityCombobox from "@/components/CityCombobox";
import { normalizePhoneToE164 } from "@/utils/phone";

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

export default function ClientDetailPage({ params }) {
  const clientId = params?.id;
  const queryClient = useQueryClient();
  const { data: user, loading: userLoading } = useUser();
  const [detailOpen, setDetailOpen] = useState(false);
  const [detailEvent, setDetailEvent] = useState(null);

  const [editOpen, setEditOpen] = useState(false);
  const [editError, setEditError] = useState(null);

  const [editType, setEditType] = useState("buyer");
  const [editName, setEditName] = useState("");
  const [editEmail, setEditEmail] = useState("");
  const [editPhone, setEditPhone] = useState("");
  const [editCountry, setEditCountry] = useState("Turkey");
  const [editCity, setEditCity] = useState("");
  const [editNotes, setEditNotes] = useState("");

  const {
    data: client,
    isLoading: clientLoading,
    error: clientError,
  } = useQuery({
    queryKey: ["client", user?.id, clientId],
    queryFn: async () => {
      const res = await fetch(
        `/api/clients?id=${encodeURIComponent(clientId)}`,
      );
      if (!res.ok) {
        throw new Error("Failed to load client");
      }
      return res.json();
    },
    enabled: !!user?.id && !!clientId,
  });

  const {
    data: shareLinks,
    isLoading: shareLinksLoading,
    error: shareLinksError,
  } = useQuery({
    queryKey: ["share-links", user?.id, clientId],
    queryFn: async () => {
      const res = await fetch(
        `/api/share-links?clientId=${encodeURIComponent(clientId)}&limit=50`,
      );
      if (!res.ok) {
        throw new Error("Failed to load share links");
      }
      return res.json();
    },
    enabled: !!user?.id && !!clientId,
  });

  const {
    data: contracts,
    isLoading: contractsLoading,
    error: contractsError,
  } = useQuery({
    queryKey: ["contracts", user?.id, clientId],
    queryFn: async () => {
      const res = await fetch(
        `/api/contracts?clientId=${encodeURIComponent(clientId)}`,
      );
      if (!res.ok) {
        throw new Error("Failed to load contracts");
      }
      return res.json();
    },
    enabled: !!user?.id && !!clientId,
  });

  const {
    events,
    upcomingEvents,
    isLoading: eventsLoading,
  } = useCalendarEvents(user?.id, { clientId });

  const pastEvents = useMemo(() => {
    const now = new Date();
    return (Array.isArray(events) ? events : [])
      .filter((e) => new Date(e.starts_at) < now)
      .sort((a, b) => new Date(b.starts_at) - new Date(a.starts_at));
  }, [events]);

  const onOpenEvent = useCallback((ev) => {
    setDetailEvent(ev);
    setDetailOpen(true);
  }, []);

  const onCloseEvent = useCallback(() => {
    setDetailEvent(null);
    setDetailOpen(false);
  }, []);

  const { countryOptions, getCities } = useCountriesAndCities();

  const cityOptions = useMemo(() => {
    return getCities(editCountry);
  }, [editCountry, getCities]);

  useEffect(() => {
    if (!client) return;
    setEditType(client.client_type || "buyer");
    setEditName(client.full_name || "");
    setEditEmail(client.email || "");
    setEditPhone(client.phone || "");
    setEditCountry(client.country || "Turkey");
    setEditCity(client.city || "");
    setEditNotes(client.notes || "");
  }, [client]);

  const updateClientMutation = useMutation({
    mutationFn: async () => {
      const safeName = editName.trim();
      if (!safeName) {
        throw new Error("Please enter the client name");
      }

      const normalizedPhone = editPhone
        ? normalizePhoneToE164(editPhone)
        : null;

      const res = await fetch(`/api/clients/${clientId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          client_type: editType,
          full_name: safeName,
          email: editEmail.trim() || null,
          phone: normalizedPhone || null,
          notes: editNotes.trim() || null,
          country: editCountry || null,
          city: editCity || null,
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not update client");
      }

      return res.json();
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: ["client", user?.id, clientId],
      });
      await queryClient.invalidateQueries({ queryKey: ["clients", user?.id] });
      setEditOpen(false);
      setEditError(null);
    },
    onError: (e) => {
      console.error(e);
      setEditError(e?.message || "Could not update client");
    },
  });

  const deleteClientMutation = useMutation({
    mutationFn: async () => {
      const ok =
        typeof window !== "undefined"
          ? window.confirm("Delete this client? This cannot be undone.")
          : false;
      if (!ok) {
        return { cancelled: true };
      }

      const res = await fetch(`/api/clients/${clientId}`, { method: "DELETE" });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not delete client");
      }
      return res.json();
    },
    onSuccess: async (data) => {
      if (data?.cancelled) return;
      await queryClient.invalidateQueries({ queryKey: ["clients", user?.id] });
      if (typeof window !== "undefined") {
        window.location.href = "/directory";
      }
    },
    onError: (e) => {
      console.error(e);
      if (typeof window !== "undefined") {
        window.alert("Could not delete this client.");
      }
    },
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

  const errMsg = clientError ? "Could not load this client." : null;

  return (
    <div className="min-h-screen ui-surface">
      <Header />

      <div className="pt-16">
        <div className="max-w-5xl mx-auto px-4 sm:px-8 py-8 sm:py-12">
          <div className="mb-6">
            <a
              href="/directory"
              className="inline-flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 font-jetbrains-mono"
            >
              <ArrowLeft size={16} />
              Back to Directory
            </a>
          </div>

          <div className="mb-8">
            <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-gray-100 mb-2 font-jetbrains-mono">
              Client
            </h1>
            <p className="text-gray-600 dark:text-gray-300 font-jetbrains-mono">
              Client profile + all appointments
            </p>
          </div>

          <div className="bg-white/70 dark:bg-white/5 rounded-2xl border border-black/10 dark:border-white/10 backdrop-blur shadow-[0_14px_60px_rgba(0,0,0,0.18)] p-6 sm:p-8">
            {clientLoading ? (
              <div className="text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                Loading client...
              </div>
            ) : errMsg ? (
              <div className="text-red-700 dark:text-red-300 font-jetbrains-mono">
                {errMsg}
              </div>
            ) : client ? (
              <div>
                <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
                  <div className="flex items-start gap-4">
                    <div className="w-12 h-12 rounded-full bg-[var(--brandSoft)] dark:bg-[var(--brandSoftDark)] flex items-center justify-center">
                      <UserIcon
                        size={22}
                        className="text-[var(--brandDark)] dark:text-[var(--brand)]"
                      />
                    </div>

                    <div className="min-w-0">
                      <div className="text-xl font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                        {client.full_name}
                      </div>

                      <div className="mt-2 flex flex-col gap-2">
                        {client.phone ? (
                          <div className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                            <Phone size={14} className="text-[var(--brand)]" />
                            <span className="truncate">{client.phone}</span>
                          </div>
                        ) : null}
                        {client.email ? (
                          <div className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                            <Mail size={14} className="text-[var(--brand)]" />
                            <span className="truncate">{client.email}</span>
                          </div>
                        ) : null}

                        {client.country || client.city ? (
                          <div className="text-xs text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                            {(client.country || "").toString()}
                            {client.city ? ` • ${client.city}` : ""}
                          </div>
                        ) : null}

                        {client.notes ? (
                          <div className="mt-3 whitespace-pre-wrap text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                            {client.notes}
                          </div>
                        ) : null}
                      </div>
                    </div>
                  </div>

                  <div className="shrink-0 flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => {
                        setEditError(null);
                        setEditOpen(true);
                      }}
                      className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-black/10 dark:border-white/10 bg-white/70 dark:bg-white/5 text-gray-900 dark:text-gray-100 text-sm font-medium hover:bg-white/80 dark:hover:bg-white/10 font-jetbrains-mono"
                    >
                      <Pencil size={16} /> Edit
                    </button>

                    <button
                      type="button"
                      onClick={() => deleteClientMutation.mutate()}
                      disabled={deleteClientMutation.isPending}
                      className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-red-200/70 dark:border-red-900/50 bg-white/70 dark:bg-white/5 text-red-700 dark:text-red-300 text-sm font-medium hover:bg-red-50 dark:hover:bg-red-900/20 font-jetbrains-mono disabled:opacity-50"
                    >
                      <Trash2 size={16} /> Delete
                    </button>
                  </div>
                </div>

                <div className="mt-8">
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                      Contracts
                    </div>
                    <a
                      href={`/contracts/new?clientId=${encodeURIComponent(clientId)}`}
                      className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-xs font-medium hover:bg-gray-800 dark:hover:bg-gray-200 font-jetbrains-mono"
                    >
                      <FileText size={14} /> Create contract
                    </a>
                  </div>

                  <div className="mt-3 rounded-xl border border-black/10 dark:border-white/10 bg-white/80 dark:bg-white/5 p-4">
                    {contractsLoading ? (
                      <div className="text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                        Loading...
                      </div>
                    ) : contractsError ? (
                      <div className="text-sm text-red-700 dark:text-red-300 font-jetbrains-mono">
                        Could not load contracts.
                      </div>
                    ) : Array.isArray(contracts) && contracts.length > 0 ? (
                      <div className="space-y-2">
                        {contracts.slice(0, 20).map((co) => {
                          const title =
                            co?.metadata?.display_name ||
                            co?.template_type ||
                            "Contract";
                          const createdAt = co?.created_at
                            ? new Date(co.created_at).toLocaleString()
                            : "";

                          return (
                            <a
                              key={co.id}
                              href={`/contracts/${co.id}`}
                              className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 rounded-lg border border-black/10 dark:border-white/10 bg-white/70 dark:bg-white/5 px-3 py-3 hover:bg-white/80 dark:hover:bg-white/10 transition-colors"
                            >
                              <div className="min-w-0">
                                <div className="text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono truncate">
                                  {title}
                                </div>
                                <div className="text-xs text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                                  {co?.property_title || "Property"}
                                  {createdAt ? ` • ${createdAt}` : ""}
                                </div>
                              </div>
                              <div className="text-xs text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                                {co?.storage_path_pdf ? "PDF ready" : "Draft"}
                              </div>
                            </a>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                        No contracts yet.
                      </div>
                    )}
                  </div>
                </div>

                <div className="mt-8">
                  <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                    Share links
                  </div>

                  <div className="mt-3 rounded-xl border border-black/10 dark:border-white/10 bg-white/80 dark:bg-white/5 p-4">
                    {shareLinksLoading ? (
                      <div className="text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                        Loading...
                      </div>
                    ) : shareLinksError ? (
                      <div className="text-sm text-red-700 dark:text-red-300 font-jetbrains-mono">
                        Could not load share links.
                      </div>
                    ) : Array.isArray(shareLinks) && shareLinks.length > 0 ? (
                      <div className="space-y-2">
                        {shareLinks.map((sl) => {
                          const origin = buildOrigin();
                          const url = sl?.slug
                            ? `${origin}/share/${sl.slug}`
                            : null;
                          const title = sl?.property_title || "Property";
                          const createdAt = sl?.created_at
                            ? new Date(sl.created_at).toLocaleString()
                            : "";

                          return (
                            <div
                              key={sl.id}
                              className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 rounded-lg border border-black/10 dark:border-white/10 bg-white/70 dark:bg-white/5 px-3 py-3"
                            >
                              <div className="min-w-0">
                                <div className="text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono truncate">
                                  {title}
                                </div>
                                <div className="text-xs text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                                  {createdAt}
                                </div>
                              </div>

                              {url ? (
                                <div className="flex items-center gap-2">
                                  <a
                                    href={url}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-xs font-medium hover:bg-gray-800 dark:hover:bg-gray-200 font-jetbrains-mono"
                                  >
                                    <LinkIcon size={14} /> Open
                                  </a>
                                  <button
                                    type="button"
                                    onClick={() => copyToClipboard(url)}
                                    className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-white dark:bg-[#262626] border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100 text-xs font-medium hover:bg-gray-50 dark:hover:bg-gray-800 font-jetbrains-mono"
                                  >
                                    <Copy size={14} /> Copy
                                  </button>
                                </div>
                              ) : null}
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div className="text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                        No share links.
                      </div>
                    )}
                  </div>
                </div>

                <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div>
                    <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                      Upcoming appointments
                    </div>
                    <div className="mt-3 space-y-3">
                      {eventsLoading ? (
                        <div className="text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                          Loading...
                        </div>
                      ) : upcomingEvents.length > 0 ? (
                        upcomingEvents.map((ev) => (
                          <EventCard
                            key={ev.id}
                            event={ev}
                            onClick={onOpenEvent}
                          />
                        ))
                      ) : (
                        <div className="text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                          No upcoming appointments.
                        </div>
                      )}
                    </div>
                  </div>

                  <div>
                    <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                      Past appointments
                    </div>
                    <div className="mt-3 space-y-3">
                      {eventsLoading ? (
                        <div className="text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                          Loading...
                        </div>
                      ) : pastEvents.length > 0 ? (
                        pastEvents.map((ev) => (
                          <EventCard
                            key={ev.id}
                            event={ev}
                            onClick={onOpenEvent}
                          />
                        ))
                      ) : (
                        <div className="text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                          No past appointments.
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                Client not found.
              </div>
            )}
          </div>
        </div>
      </div>

      {editOpen ? (
        <div className="fixed inset-0 z-[60] flex items-center justify-center p-4">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => {
              if (updateClientMutation.isPending) return;
              setEditOpen(false);
              setEditError(null);
            }}
          />
          <div className="relative w-full max-w-xl rounded-xl bg-white/95 dark:bg-[#0B0C0F]/90 border border-black/10 dark:border-white/10 shadow-xl p-6 backdrop-blur">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="text-lg font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                  Edit client
                </div>
              </div>
              <button
                type="button"
                onClick={() => {
                  if (updateClientMutation.isPending) return;
                  setEditOpen(false);
                  setEditError(null);
                }}
                className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                aria-label="Close"
              >
                <X size={20} />
              </button>
            </div>

            <div className="mt-5 grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                  Type
                </label>
                <select
                  value={editType}
                  onChange={(e) => setEditType(e.target.value)}
                  className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                >
                  <option value="owner">Owner</option>
                  <option value="buyer">Buyer</option>
                  <option value="renter">Renter</option>
                </select>
              </div>

              <div className="space-y-2">
                <label className="text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                  Name
                </label>
                <input
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                  placeholder="Full name"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                  Email (optional)
                </label>
                <input
                  value={editEmail}
                  onChange={(e) => setEditEmail(e.target.value)}
                  type="email"
                  className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                  placeholder="name@email.com"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                  Phone (optional)
                </label>
                <input
                  value={editPhone}
                  onChange={(e) => setEditPhone(e.target.value)}
                  onBlur={() => {
                    const normalized = normalizePhoneToE164(editPhone);
                    setEditPhone(normalized || "");
                  }}
                  className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                  placeholder="e.g. +90XXXXXXXXXX"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                  Country
                </label>
                <select
                  value={editCountry}
                  onChange={(e) => {
                    setEditCountry(e.target.value);
                    setEditCity("");
                  }}
                  className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                >
                  {countryOptions.length === 0 ? (
                    <option value={editCountry || ""}>Loading…</option>
                  ) : (
                    countryOptions.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))
                  )}
                </select>
              </div>

              <div className="space-y-2">
                <label className="text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                  City
                </label>
                <CityCombobox
                  value={editCity}
                  onChange={setEditCity}
                  options={cityOptions}
                  disabled={!editCountry}
                  placeholder={
                    editCountry
                      ? "Start typing a city…"
                      : "Select a country first"
                  }
                />
              </div>

              <div className="space-y-2 sm:col-span-2">
                <label className="text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                  Notes (optional)
                </label>
                <textarea
                  value={editNotes}
                  onChange={(e) => setEditNotes(e.target.value)}
                  rows={3}
                  className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                  placeholder="Optional notes"
                />
              </div>

              {editError ? (
                <div className="sm:col-span-2 text-sm text-red-600 dark:text-red-400 font-jetbrains-mono">
                  {editError}
                </div>
              ) : null}
            </div>

            <div className="mt-6 flex flex-col sm:flex-row gap-3 justify-end">
              <button
                type="button"
                onClick={() => {
                  if (updateClientMutation.isPending) return;
                  setEditOpen(false);
                  setEditError(null);
                }}
                className="px-4 py-2 rounded-lg border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100 font-jetbrains-mono"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => updateClientMutation.mutate()}
                disabled={updateClientMutation.isPending}
                className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-[var(--brand90)] hover:bg-[var(--brand)] text-white font-medium font-jetbrains-mono transition-colors disabled:opacity-50"
              >
                {updateClientMutation.isPending ? "Saving…" : "Save"}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      <EventDetailModal
        open={detailOpen}
        event={detailEvent}
        onClose={onCloseEvent}
      />
    </div>
  );
}
