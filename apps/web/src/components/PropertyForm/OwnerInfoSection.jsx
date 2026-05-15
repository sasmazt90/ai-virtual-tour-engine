import { useMemo } from "react";
import { User, ChevronDown, Search, Check, X } from "lucide-react";
import { normalizePhoneToE164_TR as normalizePhoneToE164 } from "@/utils/phone";
import { useCountriesAndCities } from "@/hooks/useCountriesAndCities";
import { TogglePill } from "./TogglePill";
import CityCombobox from "@/components/CityCombobox";

export function OwnerInfoSection({
  ownerMode,
  setOwnerMode,
  ownerSearch,
  setOwnerSearch,
  ownerDropdownOpen,
  setOwnerDropdownOpen,
  selectedOwnerClientId,
  setSelectedOwnerClientId,
  newOwnerName,
  setNewOwnerName,
  newOwnerEmail,
  setNewOwnerEmail,
  newOwnerPhone,
  setNewOwnerPhone,
  newOwnerCountryCode,
  setNewOwnerCountryCode,
  newOwnerCity,
  setNewOwnerCity,
  clients,
  clientsLoading,
  resetOwnerSelection,
}) {
  const filteredOwnerClients = useMemo(() => {
    const normalized = ownerSearch.trim().toLowerCase();
    if (!normalized) {
      return clients;
    }

    const result = clients.filter((c) => {
      const name = (c.full_name || "").toLowerCase();
      const email = (c.email || "").toLowerCase();
      const phone = (c.phone || "").toLowerCase();
      return (
        name.includes(normalized) ||
        email.includes(normalized) ||
        phone.includes(normalized)
      );
    });

    return result;
  }, [clients, ownerSearch]);

  const selectedOwnerClient = useMemo(() => {
    if (!selectedOwnerClientId) return null;
    return clients.find((c) => c.id === selectedOwnerClientId) || null;
  }, [clients, selectedOwnerClientId]);

  const { countryOptions, getCities } = useCountriesAndCities();

  const ownerCities = useMemo(() => {
    return getCities(newOwnerCountryCode);
  }, [getCities, newOwnerCountryCode]);

  const countryOptionsResolved = countryOptions;

  const ownerButtonLabel = selectedOwnerClient
    ? `${selectedOwnerClient.full_name}${selectedOwnerClient.email ? ` • ${selectedOwnerClient.email}` : ""}`
    : "Select an existing client";

  return (
    <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6 sm:p-8">
      <div className="flex items-center gap-3 mb-6">
        <User className="text-[var(--brand)]" />
        <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          Owner Info
        </h2>
      </div>

      <div className="flex flex-col sm:flex-row gap-2 mb-6">
        <TogglePill
          label="Select Existing"
          selected={ownerMode === "existing"}
          onClick={() => {
            setOwnerMode("existing");
            setNewOwnerName("");
            setNewOwnerEmail("");
            setNewOwnerPhone("");
            setNewOwnerCountryCode("Turkey");
            setNewOwnerCity("");
          }}
        />
        <TogglePill
          label="Create New Owner"
          selected={ownerMode === "new"}
          onClick={() => {
            setOwnerMode("new");
            resetOwnerSelection();
          }}
        />
      </div>

      {ownerMode === "existing" ? (
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Existing Client
          </label>

          <div className="relative">
            <button
              type="button"
              onClick={() => setOwnerDropdownOpen((v) => !v)}
              className="w-full flex items-center justify-between gap-3 px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
            >
              <span className="truncate">{ownerButtonLabel}</span>
              <ChevronDown size={18} className="text-gray-400" />
            </button>

            {ownerDropdownOpen && (
              <div className="absolute z-20 mt-2 w-full rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-[#1E1E1E] shadow-xl overflow-hidden">
                <div className="p-3 border-b border-gray-200 dark:border-gray-700">
                  <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700">
                    <Search size={16} className="text-gray-400" />
                    <input
                      value={ownerSearch}
                      onChange={(e) => setOwnerSearch(e.target.value)}
                      placeholder="Search clients..."
                      className="w-full bg-transparent outline-none text-sm text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 font-jetbrains-mono"
                    />
                    <button
                      type="button"
                      onClick={() => {
                        setOwnerDropdownOpen(false);
                      }}
                      className="text-gray-400 hover:text-gray-600"
                      aria-label="Close"
                    >
                      <X size={16} />
                    </button>
                  </div>
                </div>

                <div className="max-h-64 overflow-auto">
                  {clientsLoading ? (
                    <div className="p-4 text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                      Loading clients...
                    </div>
                  ) : filteredOwnerClients.length === 0 ? (
                    <div className="p-4 text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                      No matching clients.
                    </div>
                  ) : (
                    filteredOwnerClients.map((c) => {
                      const isSelected = selectedOwnerClientId === c.id;
                      const subtitle = c.email || c.phone || "";
                      return (
                        <button
                          key={c.id}
                          type="button"
                          onClick={() => {
                            setSelectedOwnerClientId(c.id);
                            setOwnerSearch(""); // avoid confusion: clear search after selecting
                            setOwnerDropdownOpen(false);
                          }}
                          className="w-full text-left px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                        >
                          <div className="flex items-start justify-between gap-4">
                            <div className="min-w-0">
                              <div className="font-medium text-gray-900 dark:text-gray-100 font-jetbrains-mono truncate">
                                {c.full_name}
                              </div>
                              {subtitle ? (
                                <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono truncate">
                                  {subtitle}
                                </div>
                              ) : null}
                            </div>
                            {isSelected ? (
                              <Check
                                size={18}
                                className="text-[var(--brand)] flex-shrink-0"
                              />
                            ) : null}
                          </div>
                        </button>
                      );
                    })
                  )}
                </div>
              </div>
            )}
          </div>

          {selectedOwnerClient ? (
            <div className="text-xs text-gray-600 dark:text-gray-400 font-jetbrains-mono">
              Selected: {selectedOwnerClient.full_name}
            </div>
          ) : (
            <div className="text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
              Please select an owner before saving.
            </div>
          )}

          <p className="text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
            Tip: you can also create a new owner if the client isn't in your
            directory yet.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          <div className="sm:col-span-2 space-y-2">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
              Owner Name
            </label>
            <input
              value={newOwnerName}
              onChange={(e) => setNewOwnerName(e.target.value)}
              className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
              placeholder="Full name"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
              Email (optional)
            </label>
            <input
              value={newOwnerEmail}
              onChange={(e) => setNewOwnerEmail(e.target.value)}
              type="email"
              className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
              placeholder="owner@email.com"
            />
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
              Phone (optional)
            </label>
            <input
              value={newOwnerPhone}
              onChange={(e) => setNewOwnerPhone(e.target.value)}
              onBlur={() => {
                const normalized = normalizePhoneToE164(newOwnerPhone);
                setNewOwnerPhone(normalized || "");
              }}
              className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
              placeholder="e.g. +90XXXXXXXXXX"
            />
            <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
              Include country code (E.164), e.g. +90…, +1…, +44…
            </div>
          </div>

          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
              Country
            </label>
            <select
              value={newOwnerCountryCode}
              onChange={(e) => {
                setNewOwnerCountryCode(e.target.value);
                setNewOwnerCity("");
              }}
              className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
            >
              {countryOptionsResolved.length === 0 ? (
                <option value={newOwnerCountryCode || ""}>Loading…</option>
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
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
              City
            </label>
            <CityCombobox
              value={newOwnerCity}
              onChange={setNewOwnerCity}
              options={ownerCities}
              disabled={!newOwnerCountryCode}
              placeholder={
                newOwnerCountryCode
                  ? "Start typing a city…"
                  : "Select a country first"
              }
            />
          </div>

          <div className="sm:col-span-2 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
            New owners are saved to your Directory automatically.
          </div>
        </div>
      )}
    </div>
  );
}
