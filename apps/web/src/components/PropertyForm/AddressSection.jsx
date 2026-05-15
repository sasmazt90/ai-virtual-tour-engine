import { Building2 } from "lucide-react";
import { useMemo } from "react";
import { useCountriesAndCities } from "@/hooks/useCountriesAndCities";
import CityCombobox from "@/components/CityCombobox";

export function AddressSection({
  addressLine,
  setAddressLine,
  city,
  setCity,
  postalCode,
  setPostalCode,
  country,
  setCountry,
}) {
  const { countryOptions, getCities } = useCountriesAndCities();

  const cityOptions = useMemo(() => {
    return getCities(country);
  }, [getCities, country]);

  return (
    <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6 sm:p-8">
      <div className="flex items-center gap-3 mb-6">
        <Building2 className="text-[var(--brand)]" />
        <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          Address
        </h2>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
        <div className="sm:col-span-2 space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Address line
          </label>
          <input
            value={addressLine}
            onChange={(e) => setAddressLine(e.target.value)}
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
            placeholder="Street address"
          />
        </div>

        {/* Order requested: Country, City, Postal Code */}
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Country
          </label>
          <select
            value={country}
            onChange={(e) => {
              setCountry(e.target.value);
              setCity("");
            }}
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
          >
            <option value="">Select</option>
            {countryOptions.length === 0
              ? null
              : countryOptions.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
          </select>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            City
          </label>
          <CityCombobox
            value={city}
            onChange={setCity}
            options={cityOptions}
            disabled={!country}
            placeholder={
              country ? "Start typing a city…" : "Select a country first"
            }
          />
        </div>

        <div className="space-y-2 sm:col-span-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Postal code
          </label>
          <input
            value={postalCode}
            onChange={(e) => setPostalCode(e.target.value)}
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
            placeholder="Postal code"
          />
        </div>
      </div>
    </div>
  );
}
