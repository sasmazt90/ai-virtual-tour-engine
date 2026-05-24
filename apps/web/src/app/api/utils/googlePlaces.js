function pickGoogleMapsApiKey() {
  // Best effort: if a backend-only key is later added, we can prioritize it.
  return (
    process.env.GOOGLE_MAPS_API_KEY ||
    process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY ||
    process.env.EXPO_PUBLIC_GOOGLE_MAPS_API_KEY ||
    null
  );
}

const OSM_USER_AGENT =
  process.env.OSM_USER_AGENT ||
  "360 Estate Suite/1.0 (real-estate surroundings lookup)";

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

function buildAddressString({ addressLine, city, postalCode, country }) {
  return [addressLine, city, postalCode, country]
    .map((p) => (p ? String(p).trim() : ""))
    .filter(Boolean)
    .join(", ");
}

async function geocodeViaOsm({ addressString }) {
  if (!addressString) return null;

  const url = new URL("https://nominatim.openstreetmap.org/search");
  url.searchParams.set("q", addressString);
  url.searchParams.set("format", "jsonv2");
  url.searchParams.set("limit", "1");
  url.searchParams.set("addressdetails", "1");

  let res;
  try {
    res = await fetch(url.toString(), {
      headers: {
        "User-Agent": OSM_USER_AGENT,
        Accept: "application/json",
      },
    });
  } catch (error) {
    console.warn("geocodeViaOsm: fetch failed", error);
    return null;
  }

  if (!res.ok) return null;

  let data;
  try {
    data = await res.json();
  } catch {
    return null;
  }

  const first = Array.isArray(data) ? data[0] : null;
  const lat = Number(first?.lat);
  const lng = Number(first?.lon);
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;

  return {
    ok: true,
    lat,
    lng,
    formattedAddress: first?.display_name || addressString,
  };
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
  const address = buildAddressString({ addressLine, city, postalCode, country });

  if (!address) {
    return { ok: false, error: "No address provided" };
  }

  if (!key) {
    const osm = await geocodeViaOsm({ addressString: address });
    if (osm) return osm;

    console.warn(
      "geocodeAddress: No Google Maps API key available, and OpenStreetMap fallback failed.",
    );
    return { ok: false, error: "Missing Google Maps API key" };
  }

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
    const osm = await geocodeViaOsm({ addressString: address });
    if (osm) return osm;
    return { ok: false, error: "Geocode network error" };
  }

  if (!res.ok) {
    console.warn("geocodeAddress: HTTP error", res.status, res.statusText);
    const fallback = await geocodeViaPlacesApi({ addressString: address, key });
    if (fallback) return fallback;
    const osm = await geocodeViaOsm({ addressString: address });
    if (osm) return osm;
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
      const osm = await geocodeViaOsm({ addressString: address });
      if (osm) return osm;
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

function escapeOverpassValue(value) {
  return String(value).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

function overpassStatements(filters, radiusMeters, lat, lng) {
  return filters
    .flatMap((filter) => [
      `node${filter}(around:${radiusMeters},${lat},${lng});`,
      `way${filter}(around:${radiusMeters},${lat},${lng});`,
      `relation${filter}(around:${radiusMeters},${lat},${lng});`,
    ])
    .join("\n");
}

async function osmNearbySearch({
  lat,
  lng,
  filters,
  radiusMeters = 2500,
  limit = 80,
}) {
  if (!Array.isArray(filters) || filters.length === 0) return [];

  const query = `
    [out:json][timeout:10];
    (
      ${overpassStatements(filters, radiusMeters, lat, lng)}
    );
    out center ${Number(limit) || 80};
  `;

  let res;
  try {
    res = await fetch("https://overpass-api.de/api/interpreter", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
        "User-Agent": OSM_USER_AGENT,
      },
      body: new URLSearchParams({ data: query }),
    });
  } catch (error) {
    console.warn("osmNearbySearch: fetch failed", error);
    return [];
  }

  if (!res.ok) return [];

  let data;
  try {
    data = await res.json();
  } catch (error) {
    console.warn("osmNearbySearch: failed to parse JSON", error);
    return [];
  }

  const elements = Array.isArray(data?.elements) ? data.elements : [];
  return elements
    .map((el) => {
      const elLat = Number(el?.lat ?? el?.center?.lat);
      const elLng = Number(el?.lon ?? el?.center?.lon);
      if (!Number.isFinite(elLat) || !Number.isFinite(elLng)) return null;

      const tags = el?.tags && typeof el.tags === "object" ? el.tags : {};
      const name =
        tags.name ||
        tags.brand ||
        tags.operator ||
        tags.amenity ||
        tags.shop ||
        tags.railway ||
        tags.highway ||
        null;
      const meters = haversineMeters({ lat, lng }, { lat: elLat, lng: elLng });

      return {
        place_id: `osm:${el.type}:${el.id}`,
        name,
        vicinity:
          tags["addr:street"] || tags["addr:city"] || tags.operator || null,
        types: Object.entries(tags)
          .filter(([key]) =>
            ["amenity", "shop", "railway", "highway", "station"].includes(key),
          )
          .map(([key, value]) => `${key}:${value}`),
        distance_m: Math.round(meters),
        lat: elLat,
        lng: elLng,
        tags,
      };
    })
    .filter(Boolean)
    .sort((a, b) => (a.distance_m || 0) - (b.distance_m || 0));
}

function hasType(item, value) {
  return Array.isArray(item?.types) && item.types.includes(value);
}

function hasAnyType(item, values) {
  return values.some((value) => hasType(item, value));
}

function nameMatches(item, pattern) {
  const name = String(item?.name || "");
  return pattern.test(name);
}

async function buildNearbyPlacesFromOsm({ lat, lng }) {
  const filters = [
    '["amenity"="charging_station"]',
    '["station"="subway"]',
    '["public_transport"="station"]["subway"="yes"]',
    '["railway"~"station|halt|stop"]["subway"="yes"]',
    '["railway"="station"]["subway"="yes"]',
    '["railway"="subway_entrance"]',
    '["highway"="bus_stop"]',
    '["amenity"="bus_station"]',
    '["railway"="tram_stop"]',
    '["station"="light_rail"]',
    '["amenity"="hospital"]',
    '["amenity"="clinic"]',
    '["amenity"="doctors"]',
    '["amenity"="kindergarten"]',
    '["amenity"="childcare"]',
    '["amenity"="school"]',
    '["amenity"="university"]',
    '["amenity"="college"]',
    '["shop"="mall"]',
    '["building"="retail"]',
    '["shop"="supermarket"]',
  ];

  const items = await osmNearbySearch({
    lat,
    lng,
    filters,
    radiusMeters: 7000,
    limit: 2000,
  });

  const nearest = (predicate) => items.find(predicate) || null;
  const placeRecord = (label, place) => ({
    label,
    place,
    name: place?.name || null,
    vicinity: place?.vicinity || null,
    distance_m: place?.distance_m ?? null,
    place_id: place?.place_id || null,
    types: place?.types || null,
    lat: place?.lat ?? null,
    lng: place?.lng ?? null,
  });

  const evCharging = placeRecord(
    "EV Charging Station",
    nearest((item) => hasType(item, "amenity:charging_station")),
  );
  const metroStation = placeRecord(
    "Metro Station",
    nearest(
      (item) =>
        hasType(item, "station:subway") ||
        hasType(item, "railway:subway_entrance") ||
        item?.tags?.subway === "yes",
    ),
  );
  const busStop = placeRecord(
    "Bus Stop",
    nearest((item) =>
      hasAnyType(item, ["highway:bus_stop", "amenity:bus_station"]),
    ),
  );
  const tramStation = placeRecord(
    "Tram Station",
    nearest((item) =>
      hasAnyType(item, ["railway:tram_stop", "station:light_rail"]),
    ),
  );
  const healthcare = placeRecord(
    "Hospital / Clinic / Family Doctor",
    nearest((item) =>
      hasAnyType(item, [
        "amenity:hospital",
        "amenity:clinic",
        "amenity:doctors",
      ]),
    ),
  );
  const preschool = placeRecord(
    "Kindergarten / Preschool",
    nearest((item) =>
      hasAnyType(item, ["amenity:kindergarten", "amenity:childcare"]),
    ),
  );
  const primarySchool = placeRecord(
    "Primary School",
    nearest(
      (item) =>
        hasType(item, "amenity:school") &&
        (String(item?.tags?.["isced:level"] || "").startsWith("1") ||
          nameMatches(item, /primary|grundschule/i)),
    ) || nearest((item) => hasType(item, "amenity:school")),
  );
  const secondaryCandidate =
    nearest(
      (item) =>
        hasType(item, "amenity:school") &&
        (String(item?.tags?.["isced:level"] || "").startsWith("2") ||
          nameMatches(item, /middle|secondary|mittelschule|realschule/i)),
    ) || nearest((item) => hasType(item, "amenity:school"));
  const secondarySchool = placeRecord("Middle School", secondaryCandidate);
  const highSchool = placeRecord(
    "High School",
    nearest(
      (item) =>
        hasType(item, "amenity:school") &&
        (String(item?.tags?.["isced:level"] || "").startsWith("3") ||
          nameMatches(item, /high school|gymnasium|lyceum/i)),
    ) || secondaryCandidate,
  );
  const university = placeRecord(
    "University",
    nearest((item) => hasAnyType(item, ["amenity:university", "amenity:college"])),
  );
  const shoppingMall = placeRecord(
    "Shopping Mall",
    nearest((item) => hasAnyType(item, ["shop:mall", "building:retail"])),
  );
  const supermarket = placeRecord(
    "Chain Supermarket",
    nearest((item) => hasType(item, "shop:supermarket")),
  );

  return {
    surroundings: {
      transportation: [evCharging, metroStation, busStop, tramStation],
      healthcare: [healthcare],
      education: [
        preschool,
        primarySchool,
        secondarySchool,
        highSchool,
        university,
      ],
      shopping: [shoppingMall, supermarket],
    },
    transport: [evCharging, metroStation, busStop, tramStation].flatMap((item) =>
      item?.place ? [item.place] : [],
    ),
    health: healthcare?.place ? [healthcare.place] : [],
    education: [
      preschool,
      primarySchool,
      secondarySchool,
      highSchool,
      university,
    ].flatMap((item) => (item?.place ? [item.place] : [])),
    shopping: [shoppingMall, supermarket].flatMap((item) =>
      item?.place ? [item.place] : [],
    ),
    computed_at: new Date().toISOString(),
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

function mergeTop(arrays, limit = 8) {
  const seen = new Set();
  const merged = [];

  for (const arr of arrays) {
    for (const item of Array.isArray(arr) ? arr : []) {
      const key = item.place_id || `${item.name}-${item.lat}-${item.lng}`;
      if (seen.has(key)) continue;
      seen.add(key);
      merged.push(item);
    }
  }

  merged.sort((a, b) => (a.distance_m || 0) - (b.distance_m || 0));
  return merged.slice(0, limit);
}

function nearestFrom(arrays) {
  return mergeTop(arrays, 1)[0] || null;
}

async function nearestPlace({ lat, lng, label, queries, radiusMeters = 3000 }) {
  const hasGoogleKey = Boolean(pickGoogleMapsApiKey());
  const results = await Promise.all(
    queries.map((query) => {
      if (query.osm) {
        if (hasGoogleKey) return [];

        return osmNearbySearch({
          lat,
          lng,
          radiusMeters,
          filters: query.osm,
        });
      }

      return nearbySearch({
        lat,
        lng,
        radiusMeters,
        type: query.type,
        keyword: query.keyword,
      });
    }),
  );

  const place = nearestFrom(results);
  return {
    label,
    place,
    name: place?.name || null,
    vicinity: place?.vicinity || null,
    distance_m: place?.distance_m ?? null,
    place_id: place?.place_id || null,
    types: place?.types || null,
    lat: place?.lat ?? null,
    lng: place?.lng ?? null,
  };
}

export async function buildNearbyPlaces({ lat, lng }) {
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) {
    return null;
  }

  try {
    if (!pickGoogleMapsApiKey()) {
      return await buildNearbyPlacesFromOsm({ lat, lng });
    }

    const [
      evCharging,
      metroStation,
      busStop,
      tramStation,
      healthcare,
      preschool,
      primarySchool,
      secondarySchool,
      highSchool,
      university,
      shoppingMall,
      supermarket,
    ] = await Promise.all([
      nearestPlace({
        lat,
        lng,
        label: "EV Charging Station",
        queries: [
          { type: "electric_vehicle_charging_station" },
          { keyword: "electric vehicle charging station" },
          { osm: ['["amenity"="charging_station"]'] },
        ],
      }),
      nearestPlace({
        lat,
        lng,
        label: "Metro Station",
        queries: [
          { type: "subway_station" },
          { keyword: "metro station" },
          {
            osm: [
              '["station"="subway"]',
              '["public_transport"="station"]["subway"="yes"]',
              '["railway"~"station|halt|stop"]["subway"="yes"]',
              '["railway"="station"]["subway"="yes"]',
              '["railway"="subway_entrance"]',
            ],
          },
        ],
      }),
      nearestPlace({
        lat,
        lng,
        label: "Bus Stop",
        queries: [
          { type: "bus_station" },
          { type: "transit_station", keyword: "bus stop" },
          { keyword: "bus stop" },
          { osm: ['["highway"="bus_stop"]', '["amenity"="bus_station"]'] },
        ],
      }),
      nearestPlace({
        lat,
        lng,
        label: "Tram Station",
        queries: [
          { type: "light_rail_station" },
          { type: "transit_station", keyword: "tram station" },
          { keyword: "tram station" },
          { osm: ['["railway"="tram_stop"]', '["station"="light_rail"]'] },
        ],
      }),
      nearestPlace({
        lat,
        lng,
        label: "Hospital / Clinic / Family Doctor",
        queries: [
          { type: "hospital" },
          { type: "doctor" },
          { type: "health", keyword: "clinic" },
          { keyword: "family doctor" },
          {
            osm: [
              '["amenity"="hospital"]',
              '["amenity"="clinic"]',
              '["amenity"="doctors"]',
            ],
          },
        ],
        radiusMeters: 3500,
      }),
      nearestPlace({
        lat,
        lng,
        label: "Kindergarten / Preschool",
        queries: [
          { type: "preschool" },
          { keyword: "kindergarten" },
          { keyword: "preschool" },
          { keyword: "daycare" },
          { osm: ['["amenity"="kindergarten"]', '["amenity"="childcare"]'] },
        ],
        radiusMeters: 3500,
      }),
      nearestPlace({
        lat,
        lng,
        label: "Primary School",
        queries: [
          { type: "primary_school" },
          { keyword: "primary school" },
          {
            osm: [
              '["amenity"="school"]["isced:level"~"^1"]',
              `["amenity"="school"]["name"~"${escapeOverpassValue("primary|grundschule")}",i]`,
              '["amenity"="school"]',
            ],
          },
        ],
        radiusMeters: 3500,
      }),
      nearestPlace({
        lat,
        lng,
        label: "Middle School",
        queries: [
          { type: "secondary_school" },
          { keyword: "middle school" },
          { keyword: "secondary school" },
          {
            osm: [
              '["amenity"="school"]["isced:level"~"^2"]',
              `["amenity"="school"]["name"~"${escapeOverpassValue("middle|secondary|mittelschule|realschule")}",i]`,
              '["amenity"="school"]',
            ],
          },
        ],
        radiusMeters: 4500,
      }),
      nearestPlace({
        lat,
        lng,
        label: "High School",
        queries: [
          { type: "secondary_school", keyword: "high school" },
          { keyword: "high school" },
          {
            osm: [
              '["amenity"="school"]["isced:level"~"^3"]',
              `["amenity"="school"]["name"~"${escapeOverpassValue("high school|gymnasium|lyceum")}",i]`,
              '["amenity"="school"]',
            ],
          },
        ],
        radiusMeters: 5000,
      }),
      nearestPlace({
        lat,
        lng,
        label: "University",
        queries: [
          { type: "university" },
          { osm: ['["amenity"="university"]', '["amenity"="college"]'] },
        ],
        radiusMeters: 7000,
      }),
      nearestPlace({
        lat,
        lng,
        label: "Shopping Mall",
        queries: [
          { type: "shopping_mall" },
          { osm: ['["shop"="mall"]', '["building"="retail"]'] },
        ],
        radiusMeters: 5000,
      }),
      nearestPlace({
        lat,
        lng,
        label: "Chain Supermarket",
        queries: [
          { type: "supermarket" },
          { keyword: "supermarket" },
          { keyword: "grocery store" },
          { osm: ['["shop"="supermarket"]'] },
        ],
        radiusMeters: 3000,
      }),
    ]);

    return {
      surroundings: {
        transportation: [evCharging, metroStation, busStop, tramStation],
        healthcare: [healthcare],
        education: [
          preschool,
          primarySchool,
          secondarySchool,
          highSchool,
          university,
        ],
        shopping: [shoppingMall, supermarket],
      },
      // Backward-compatible grouped lists used by older UI.
      transport: [
        evCharging,
        metroStation,
        busStop,
        tramStation,
      ].flatMap((item) => (item?.place ? [item.place] : [])),
      health: healthcare?.place ? [healthcare.place] : [],
      education: [
        preschool,
        primarySchool,
        secondarySchool,
        highSchool,
        university,
      ].flatMap((item) => (item?.place ? [item.place] : [])),
      shopping: [shoppingMall, supermarket].flatMap((item) =>
        item?.place ? [item.place] : [],
      ),
      computed_at: new Date().toISOString(),
    };
  } catch (error) {
    // Best-effort only.
    console.warn("buildNearbyPlaces: failed", error);
    return null;
  }
}
