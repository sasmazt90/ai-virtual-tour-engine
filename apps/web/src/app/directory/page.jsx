import { Header } from "../../components/Header";
import { useCallback, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import useUser from "@/utils/useUser";
import { Plus, User, Mail, Phone, Building, X } from "lucide-react";
import { useCountriesAndCities } from "@/hooks/useCountriesAndCities";
import { normalizePhoneToE164 } from "@/utils/phone";
import CityCombobox from "@/components/CityCombobox";

export default function DirectoryPage() {
  const queryClient = useQueryClient();
  const { data: user, loading: userLoading } = useUser();
  const [typeFilter, setTypeFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [showAddModal, setShowAddModal] = useState(false);
  const [addError, setAddError] = useState(null);

  const [newType, setNewType] = useState("buyer");
  const [newName, setNewName] = useState("");
  const [newEmail, setNewEmail] = useState("");
  const [newPhone, setNewPhone] = useState("");
  const [newCountryCode, setNewCountryCode] = useState("Turkey");
  const [newCity, setNewCity] = useState("");
  const [newNotes, setNewNotes] = useState("");

  const { data: clients, isLoading } = useQuery({
    queryKey: ["clients", user?.id, typeFilter, search],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (typeFilter !== "all") params.append("type", typeFilter);
      if (search) params.append("search", search);

      const res = await fetch(`/api/clients?${params.toString()}`);
      if (!res.ok) throw new Error("Failed to fetch clients");
      return res.json();
    },
    enabled: !!user?.id,
  });

  const { countryOptions, getCities } = useCountriesAndCities();

  const cityOptions = useMemo(() => {
    return getCities(newCountryCode);
  }, [getCities, newCountryCode]);

  const addClientMutation = useMutation({
    mutationFn: async (payload) => {
      const res = await fetch("/api/clients", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to add client");
      }
      return res.json();
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["clients", user?.id] });
      setShowAddModal(false);
      setAddError(null);

      setNewType("buyer");
      setNewName("");
      setNewEmail("");
      setNewPhone("");
      setNewCountryCode("Turkey");
      setNewCity("");
      setNewNotes("");
    },
    onError: (e) => {
      console.error(e);
      setAddError(e?.message || "Failed to add client");
    },
  });

  const openAddModal = useCallback(() => {
    setAddError(null);
    setShowAddModal(true);
  }, []);

  const closeAddModal = useCallback(() => {
    setShowAddModal(false);
    setAddError(null);
  }, []);

  const onSubmitAddClient = useCallback(() => {
    setAddError(null);

    const safeName = newName.trim();
    if (!safeName) {
      setAddError("Please enter the client name");
      return;
    }

    const normalizedPhone = newPhone ? normalizePhoneToE164(newPhone) : null;

    addClientMutation.mutate({
      client_type: newType,
      full_name: safeName,
      email: newEmail.trim() || null,
      phone: normalizedPhone || null,
      notes: newNotes.trim() || null,
      country: newCountryCode || null,
      city: newCity || null,
    });
  }, [
    addClientMutation,
    newCity,
    newCountryCode,
    newEmail,
    newName,
    newNotes,
    newPhone,
    newType,
  ]);

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

  const countryOptionsResolved = countryOptions;

  return (
    <div className="min-h-screen ui-surface">
      <Header />

      <div className="pt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-8 py-8 sm:py-12">
          <div className="mb-8 flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4">
            <div>
              <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-gray-100 mb-2 font-jetbrains-mono">
                Directory
              </h1>
              <p className="text-gray-600 dark:text-gray-300 font-jetbrains-mono">
                Manage your clients
              </p>
            </div>

            <button
              onClick={openAddModal}
              className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-[var(--brand90)] hover:bg-[var(--brand)] text-white font-medium font-jetbrains-mono transition-colors"
            >
              <Plus size={18} />
              Add client
            </button>
          </div>

          <div className="mb-8 flex flex-col sm:flex-row gap-4">
            <div className="flex-1">
              <input
                type="text"
                placeholder="Search clients..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full px-4 py-3 bg-white dark:bg-[#262626] border border-gray-200 dark:border-gray-700 rounded-lg text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
              />
            </div>
            <div className="flex gap-2 flex-wrap">
              <button
                onClick={() => setTypeFilter("all")}
                className={`px-4 py-2 rounded-lg font-medium transition-colors font-jetbrains-mono ${
                  typeFilter === "all"
                    ? "bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900"
                    : "bg-white dark:bg-[#262626] text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-700"
                }`}
              >
                All
              </button>
              <button
                onClick={() => setTypeFilter("owner")}
                className={`px-4 py-2 rounded-lg font-medium transition-colors font-jetbrains-mono ${
                  typeFilter === "owner"
                    ? "bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900"
                    : "bg-white dark:bg-[#262626] text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-700"
                }`}
              >
                Owners
              </button>
              <button
                onClick={() => setTypeFilter("buyer")}
                className={`px-4 py-2 rounded-lg font-medium transition-colors font-jetbrains-mono ${
                  typeFilter === "buyer"
                    ? "bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900"
                    : "bg-white dark:bg-[#262626] text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-700"
                }`}
              >
                Buyers
              </button>
              <button
                onClick={() => setTypeFilter("renter")}
                className={`px-4 py-2 rounded-lg font-medium transition-colors font-jetbrains-mono ${
                  typeFilter === "renter"
                    ? "bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900"
                    : "bg-white dark:bg-[#262626] text-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-700"
                }`}
              >
                Renters
              </button>
            </div>
          </div>

          {isLoading ? (
            <div className="text-center py-12">
              <p className="text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                Loading clients...
              </p>
            </div>
          ) : clients && clients.length > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {clients.map((client) => {
                const typeLabel =
                  client.client_type.charAt(0).toUpperCase() +
                  client.client_type.slice(1);

                return (
                  <a
                    key={client.id}
                    href={`/directory/${client.id}`}
                    className="block bg-white/70 dark:bg-white/5 rounded-xl border border-black/10 dark:border-white/10 backdrop-blur shadow-[0_14px_60px_rgba(0,0,0,0.18)] p-6 hover:bg-white/80 dark:hover:bg-white/10 transition-colors"
                  >
                    <div className="flex items-start justify-between mb-4">
                      <div className="w-12 h-12 bg-[var(--brandSoft)] dark:bg-[var(--brandSoftDark)] rounded-full flex items-center justify-center">
                        <User
                          className="text-[var(--brandDark)] dark:text-[var(--brand)]"
                          size={24}
                        />
                      </div>
                      <span
                        className={`px-2 py-1 text-xs rounded-full font-medium ${
                          client.client_type === "owner"
                            ? "bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200"
                            : client.client_type === "buyer"
                              ? "bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200"
                              : "bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200"
                        }`}
                      >
                        {typeLabel}
                      </span>
                    </div>

                    <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-3 font-jetbrains-mono">
                      {client.full_name}
                    </h3>

                    <div className="space-y-2">
                      {client.email ? (
                        <div className="flex items-center text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                          <Mail
                            size={14}
                            className="mr-2 text-[var(--brand)]"
                          />
                          <span className="truncate">{client.email}</span>
                        </div>
                      ) : null}

                      {client.phone ? (
                        <div className="flex items-center text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                          <Phone
                            size={14}
                            className="mr-2 text-[var(--brand)]"
                          />
                          <span>{client.phone}</span>
                        </div>
                      ) : null}

                      {client.country || client.city ? (
                        <div className="text-xs text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                          {(client.country || "").toString()}
                          {client.city ? ` • ${client.city}` : ""}
                        </div>
                      ) : null}

                      {client.notes ? (
                        <p className="text-sm text-gray-600 dark:text-gray-400 mt-3 line-clamp-2 font-jetbrains-mono">
                          {client.notes}
                        </p>
                      ) : null}
                    </div>
                  </a>
                );
              })}
            </div>
          ) : (
            <div className="text-center py-12 bg-white/70 dark:bg-white/5 border border-black/10 dark:border-white/10 rounded-xl backdrop-blur">
              <Building className="mx-auto mb-4 text-gray-400" size={48} />
              <p className="text-gray-600 dark:text-gray-400 mb-4 font-jetbrains-mono">
                No clients yet
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-500 font-jetbrains-mono">
                Clients will be added when you create properties with owner
                information.
              </p>
            </div>
          )}
        </div>
      </div>

      {showAddModal ? (
        <div className="fixed inset-0 z-[60] flex items-center justify-center p-4">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={closeAddModal}
          />
          <div className="relative w-full max-w-xl rounded-xl bg-white/95 dark:bg-[#0B0C0F]/90 border border-black/10 dark:border-white/10 shadow-xl p-6 backdrop-blur">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="text-lg font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                  Add client
                </div>
              </div>
              <button
                type="button"
                onClick={closeAddModal}
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
                  value={newType}
                  onChange={(e) => setNewType(e.target.value)}
                  className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                >
                  <option value="owner">Owner</option>
                  <option value="buyer">Buyer</option>
                  <option value="renter">Renter</option>
                </select>
              </div>

              <div className="space-y-2 sm:col-span-1">
                <label className="text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                  Name
                </label>
                <input
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                  placeholder="Full name"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                  Email (optional)
                </label>
                <input
                  value={newEmail}
                  onChange={(e) => setNewEmail(e.target.value)}
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
                  value={newPhone}
                  onChange={(e) => setNewPhone(e.target.value)}
                  onBlur={() => {
                    const normalized = normalizePhoneToE164(newPhone);
                    setNewPhone(normalized || "");
                  }}
                  className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                  placeholder="e.g. +90XXXXXXXXXX"
                />
                <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
                  Include country code (E.164), e.g. +90…, +1…, +44…
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                  Country
                </label>
                <select
                  value={newCountryCode}
                  onChange={(e) => {
                    setNewCountryCode(e.target.value);
                    setNewCity("");
                  }}
                  className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                >
                  {countryOptionsResolved.length === 0 ? (
                    <option value={newCountryCode || ""}>Loading…</option>
                  ) : (
                    countryOptionsResolved.map((c) => (
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
                  value={newCity}
                  onChange={setNewCity}
                  options={cityOptions}
                  disabled={!newCountryCode}
                  placeholder={
                    newCountryCode
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
                  value={newNotes}
                  onChange={(e) => setNewNotes(e.target.value)}
                  rows={3}
                  className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                  placeholder="Optional notes"
                />
              </div>

              {addError ? (
                <div className="sm:col-span-2 text-sm text-red-600 dark:text-red-400 font-jetbrains-mono">
                  {addError}
                </div>
              ) : null}
            </div>

            <div className="mt-6 flex flex-col sm:flex-row gap-3 justify-end">
              <button
                type="button"
                onClick={closeAddModal}
                className="px-4 py-2 rounded-lg border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100 font-jetbrains-mono"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={onSubmitAddClient}
                disabled={addClientMutation.isPending}
                className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-[var(--brand90)] hover:bg-[var(--brand)] text-white font-medium font-jetbrains-mono transition-colors disabled:opacity-50"
              >
                {addClientMutation.isPending ? "Saving…" : "Save"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
