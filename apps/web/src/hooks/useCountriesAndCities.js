import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  getCityOverrideForCountry,
  normalizeCountryKey,
  resolveCanonicalCountryName,
} from "@/utils/locations";

// Public dataset (no backend required). Returns many countries and their cities.
// Docs/Source: countriesnow.space
const COUNTRIESNOW_URL = "https://countriesnow.space/api/v0.1/countries";

const REQUIRED_COUNTRIES = [
  "Turkey",
  "United Arab Emirates",
  "India",
  "Russia",
  "United States",
  "United Kingdom",
  "Canada",
].map((c) => c.toLowerCase());

function safeString(v) {
  if (v === null || v === undefined) return "";
  return String(v).trim();
}

function sortAlpha(arr) {
  return arr.slice().sort((a, b) => a.localeCompare(b, "en"));
}

function normalizeCityList(countryLike, rawCities) {
  const override = getCityOverrideForCountry(countryLike);
  if (Array.isArray(override) && override.length > 0) {
    return sortAlpha(
      Array.from(new Set(override.map((c) => safeString(c)).filter(Boolean))),
    );
  }

  const list = Array.isArray(rawCities)
    ? rawCities.map((c) => safeString(c)).filter(Boolean)
    : [];

  // De-dupe. (We avoid rendering all cities at once in the UI; the UI filters as you type.)
  const unique = Array.from(new Set(list));

  // Sorting helps the UI, but can be expensive for extremely large lists.
  // Keep it simple: sort up to a reasonable size; otherwise return the unsorted unique list.
  const shouldSort = unique.length <= 5000;
  return shouldSort ? sortAlpha(unique) : unique;
}

export function useCountriesAndCities() {
  const query = useQuery({
    queryKey: ["countriesAndCities"],
    queryFn: async () => {
      const res = await fetch(COUNTRIESNOW_URL);
      if (!res.ok) {
        throw new Error(
          `When fetching ${COUNTRIESNOW_URL}, the response was [${res.status}] ${res.statusText}`,
        );
      }

      const json = await res.json().catch(() => ({}));
      const rows = Array.isArray(json?.data) ? json.data : [];

      const countries = [];
      const rawCitiesByCountry = new Map();

      for (const row of rows) {
        const country = safeString(row?.country);
        const cities = Array.isArray(row?.cities) ? row.cities : [];

        if (!country) continue;

        countries.push(country);
        rawCitiesByCountry.set(country, cities);
      }

      const uniqueCountries = sortAlpha(Array.from(new Set(countries)));

      // Fast lookup: normalized country key => upstream country name.
      const normalizedToUpstream = new Map();
      for (const c of uniqueCountries) {
        const k = normalizeCountryKey(c);
        if (!k) continue;
        if (!normalizedToUpstream.has(k)) {
          normalizedToUpstream.set(k, c);
        }
      }

      // Sanity: make sure required countries appear (even if the upstream data changes).
      const hasCountry = new Set(uniqueCountries.map((c) => c.toLowerCase()));
      const missing = REQUIRED_COUNTRIES.filter((c) => !hasCountry.has(c));

      if (missing.length > 0) {
        for (const m of missing) {
          const title = m
            .split(" ")
            .map((p) => p.charAt(0).toUpperCase() + p.slice(1))
            .join(" ");
          uniqueCountries.push(title);
          rawCitiesByCountry.set(title, []);

          const key = normalizeCountryKey(title);
          if (key && !normalizedToUpstream.has(key)) {
            normalizedToUpstream.set(key, title);
          }
        }
      }

      return {
        countries: sortAlpha(uniqueCountries),
        rawCitiesByCountry,
        normalizedToUpstream,
      };
    },
    staleTime: 1000 * 60 * 60 * 24, // 24h
    retry: 1,
  });

  const countryOptions = useMemo(() => {
    return Array.isArray(query.data?.countries) ? query.data.countries : [];
  }, [query.data?.countries]);

  const getCities = useMemo(() => {
    const map = query.data?.rawCitiesByCountry;
    const normalizedToUpstream = query.data?.normalizedToUpstream;
    const cache = new Map();

    return (countryLike) => {
      if (!countryLike || !map) return [];

      // Use canonical country name for caching (so UAE / United Arab Emirates don't duplicate).
      const canonical =
        resolveCanonicalCountryName(countryLike) || safeString(countryLike);

      if (cache.has(canonical)) {
        return cache.get(canonical);
      }

      // First try direct key (upstream name).
      let rawCities = map.get(countryLike);

      // If missing, try canonical name.
      if (!rawCities) {
        rawCities = map.get(canonical);
      }

      // If still missing, try normalized lookup against upstream dataset.
      if (!rawCities && normalizedToUpstream) {
        const norm = normalizeCountryKey(countryLike);
        const upstreamName = norm ? normalizedToUpstream.get(norm) : null;
        if (upstreamName) {
          rawCities = map.get(upstreamName);
        }
      }

      const normalizedCities = normalizeCityList(canonical, rawCities);
      cache.set(canonical, normalizedCities);
      return normalizedCities;
    };
  }, [query.data?.rawCitiesByCountry, query.data?.normalizedToUpstream]);

  return {
    countryOptions,
    getCities,
    isLoading: query.isLoading,
    error: query.error,
    refetch: query.refetch,
  };
}
