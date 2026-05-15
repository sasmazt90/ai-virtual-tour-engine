function pickGoogleMapsApiKey() {
  // Best effort: if a backend-only key is later added, we can prioritize it.
  return (
    process.env.GOOGLE_MAPS_API_KEY ||
    process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY ||
    process.env.EXPO_PUBLIC_GOOGLE_MAPS_API_KEY ||
    null
  );
}

function toRad(x) {
  return (x * Math.PI) / 180;
}

function haversineMeters(a, b) {
  const R = 6371000;
  const dLat = toRad(b.lat - a.lat);
  const dLng = toRad(b.lng - a.lng);
  const lat1 = toRad(a.lat);
  const lat2 = toRad(b.lat);
  const sinDLat = Math.sin(dLat / 2);
  const sinDLng = Math.sin(dLng / 2);

  const h =
    sinDLat * sinDLat + Math.cos(lat1) * Math.cos(lat2) * sinDLng * sinDLng;
  const c = 2 * Math.atan2(Math.sqrt(h), Math.sqrt(1 - h));
  return R * c;
}

// Fallback: use Places API findplacefromtext to get coordinates when Geocoding API is not enabled
async function geocodeViaPlacesApi({ addressString, key }) {
  const url = new URL(
    "https://maps.googleapis.com/maps/api/place/findplacefromtext/json",
  );
  url.searchParams.set("input", addressString);
  url.searchParams.set("inputtype", "textquery");
  url.searchParams.set("fields", "geometry,formatted_address");
  url.searchParams.set("key", key);

  let res;
  try {
    res = await fetch(url.toString());
  } catch (error) {
    console.warn("geocodeViaPlacesApi: fetch failed", error);
    return null;
  }

  if (!res.ok) return null;

  let data;
  try {
    data = await res.json();
  } catch {
    return null;
  }

  if (data?.status !== "OK") {
    console.warn(
      "geocodeViaPlacesApi: status =",
      data?.status,
      data?.error_message,
    );
    return null;
  }

  const first = Array.isArray(data?.candidates) ? data.candidates[0] : null;
  const loc = first?.geometry?.location;
  const lat = Number(loc?.lat);
  const lng = Number(loc?.lng);

  if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;

  return {
    ok: true,
    lat,
    lng,
    formattedAddress: first?.formatted_address || addressString,
  };
}

export async function geocodeAddress({
  addressLine,
  city,
  postalCode,
  country,
}) {
  const key = pickGoogleMapsApiKey();
  if (!key) {
    console.warn(
      "geocodeAddress: No Google Maps API key available. Set GOOGLE_MAPS_API_KEY, NEXT_PUBLIC_GOOGLE_MAPS_API_KEY, or EXPO_PUBLIC_GOOGLE_MAPS_API_KEY.",
    );
    return { ok: false, error: "Missing Google Maps API key" };
  }

  const parts = [addressLine, city, postalCode, country]
    .map((p) => (p ? String(p).trim() : ""))
    .filter(Boolean);

  if (parts.length === 0) {
    return { ok: false, error: "No address provided" };
  }

  const address = parts.join(", ");

  // Try standard Geocoding API first
  const url = new URL("https://maps.googleapis.com/maps/api/geocode/json");
  url.searchParams.set("address", address);
  url.searchParams.set("key", key);

  let res;
  try {
    res = await fetch(url.toString());
  } catch (error) {
    console.warn("geocodeAddress: fetch failed", error);
    // Try Places fallback
    const fallback = await geocodeViaPlacesApi({ addressString: address, key });
    if (fallback) return fallback;
    return { ok: false, error: "Geocode network error" };
  }

  if (!res.ok) {
    console.warn("geocodeAddress: HTTP error", res.status, res.statusText);
    const fallback = await geocodeViaPlacesApi({ addressString: address, key });
    if (fallback) return fallback;
    return {
      ok: false,
      error: `Geocode failed: [${res.status}] ${res.statusText}`,
    };
  }

  let data;
  try {
    data = await res.json();
  } catch (error) {
    console.warn("geocodeAddress: failed to parse JSON", error);
    return { ok: false, error: "Geocode parse error" };
  }

  // Google can return 200 with an error in the payload.
  if (data?.status && data.status !== "OK") {
    const msg = data?.error_message || data?.status || "Geocode failed";
    console.warn(
      "geocodeAddress: Google API returned status:",
      data.status,
      "message:",
      msg,
    );
    // If Geocoding API is not enabled (REQUEST_DENIED), try Places API fallback
    if (
      data.status === "REQUEST_DENIED" ||
      data.status === "OVER_QUERY_LIMIT"
    ) {
      const fallback = await geocodeViaPlacesApi({
        addressString: address,
        key,
      });
      if (fallback) return fallback;
    }
    return { ok: false, error: msg };
  }

  const first = Array.isArray(data?.results) ? data.results[0] : null;
  const loc = first?.geometry?.location;

  const lat = Number(loc?.lat);
  const lng = Number(loc?.lng);

  if (!Number.isFinite(lat) || !Number.isFinite(lng)) {
    return { ok: false, error: "Could not geocode address" };
  }

  return {
    ok: true,
    lat,
    lng,
    formattedAddress: first?.formatted_address || address,
  };
}

async function nearbySearch({ lat, lng, keyword, type, radiusMeters = 2500 }) {
  const key = pickGoogleMapsApiKey();
  if (!key) return [];

  const url = new URL(
    "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
  );
  url.searchParams.set("location", `${lat},${lng}`);
  url.searchParams.set("radius", String(radiusMeters));
  if (type) url.searchParams.set("type", type);
  if (keyword) url.searchParams.set("keyword", keyword);
  url.searchParams.set("key", key);

  let res;
  try {
    res = await fetch(url.toString());
  } catch (error) {
    console.warn("nearbySearch: fetch failed", error);
    return [];
  }

  if (!res.ok) {
    return [];
  }

  let data;
  try {
    data = await res.json();
  } catch (error) {
    console.warn("nearbySearch: failed to parse JSON", error);
    return [];
  }

  if (data?.status && data.status !== "OK") {
    return [];
  }

  const results = Array.isArray(data?.results) ? data.results : [];

  return results
    .map((r) => {
      const rLat = Number(r?.geometry?.location?.lat);
      const rLng = Number(r?.geometry?.location?.lng);
      if (!Number.isFinite(rLat) || !Number.isFinite(rLng)) return null;

      const meters = haversineMeters({ lat, lng }, { lat: rLat, lng: rLng });

      return {
        place_id: r?.place_id || null,
        name: r?.name || null,
        vicinity: r?.vicinity || null,
        types: r?.types || null,
        distance_m: Math.round(meters),
        lat: rLat,
        lng: rLng,
      };
    })
    .filter(Boolean)
    .sort((a, b) => (a.distance_m || 0) - (b.distance_m || 0));
}

export async function buildNearbyPlaces({ lat, lng }) {
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) {
    return null;
  }

  try {
    // NOTE: Google Places Nearby Search type list is limited.
    // We use best-effort types + keyword fallbacks.
    const [
      transportA,
      transportB,
      healthA,
      healthB,
      educationA,
      educationB,
      shoppingA,
      shoppingB,
    ] = await Promise.all([
      nearbySearch({ lat, lng, type: "subway_station" }),
      nearbySearch({ lat, lng, type: "bus_station" }),
      nearbySearch({ lat, lng, type: "hospital" }),
      nearbySearch({ lat, lng, type: "veterinary_care" }),
      nearbySearch({ lat, lng, type: "school" }),
      nearbySearch({ lat, lng, type: "university" }),
      nearbySearch({ lat, lng, type: "shopping_mall" }),
      nearbySearch({ lat, lng, type: "supermarket" }),
    ]);

    const mergeTop = (arrays, limit = 8) => {
      const seen = new Set();
      const merged = [];

      for (const arr of arrays) {
        for (const item of arr) {
          const key = item.place_id || `${item.name}-${item.lat}-${item.lng}`;
          if (seen.has(key)) continue;
          seen.add(key);
          merged.push(item);
        }
      }

      merged.sort((a, b) => (a.distance_m || 0) - (b.distance_m || 0));
      return merged.slice(0, limit);
    };

    return {
      transport: mergeTop([transportA, transportB], 10),
      health: mergeTop([healthA, healthB], 10),
      education: mergeTop([educationA, educationB], 10),
      shopping: mergeTop([shoppingA, shoppingB], 10),
      computed_at: new Date().toISOString(),
    };
  } catch (error) {
    // Best-effort only.
    console.warn("buildNearbyPlaces: failed", error);
    return null;
  }
}
