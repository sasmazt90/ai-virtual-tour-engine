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
const OVERPASS_ENDPOINTS = [
  "https://overpass-api.de/api/interpreter",
  "https://overpass.osm.ch/api/interpreter",
];
const OSM_FETCH_TIMEOUT_MS = 15000;

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
  const useGoogle = process.env.ENABLE_GOOGLE_PLACES_LOOKUP === "true";
  const key = useGoogle ? pickGoogleMapsApiKey() : null;
  const address = buildAddressString({ addressLine, city, postalCode, country });

  if (!address) {
    return { ok: false, error: "No address provided" };
  }

  if (!key) {
    const osm = await geocodeViaOsm({ addressString: address });
    if (osm) return osm;

    console.warn(
      "geocodeAddress: OpenStreetMap geocoding failed.",
    );
    return { ok: false, error: "Could not geocode address with OpenStreetMap" };
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

  let data = null;
  for (const endpoint of OVERPASS_ENDPOINTS) {
    let res;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), OSM_FETCH_TIMEOUT_MS);
    try {
      res = await fetch(endpoint, {
        method: "POST",
        signal: controller.signal,
        headers: {
          "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
          "User-Agent": OSM_USER_AGENT,
        },
        body: new URLSearchParams({ data: query }),
      });
    } catch (error) {
      console.warn("osmNearbySearch: fetch failed", endpoint, error);
      continue;
    } finally {
      clearTimeout(timeoutId);
    }

    if (!res.ok) continue;

    try {
      data = await res.json();
    } catch (error) {
      console.warn("osmNearbySearch: failed to parse JSON", endpoint, error);
      continue;
    }

    if (Array.isArray(data?.elements)) break;
  }

  if (!Array.isArray(data?.elements)) return [];

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
            [
              "amenity",
              "building",
              "healthcare",
              "public_transport",
              "railway",
              "shop",
              "station",
              "highway",
            ].includes(key),
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

function uniqueTop(items, predicate, limit = 3) {
  const seen = new Set();
  const matches = [];

  for (const item of Array.isArray(items) ? items : []) {
    if (predicate && !predicate(item)) continue;
    const key = item.place_id || `${item.name}-${item.lat}-${item.lng}`;
    if (!key || seen.has(key)) continue;
    seen.add(key);
    matches.push(item);
    if (matches.length >= limit) break;
  }

  return matches;
}

function placeRecord(label, place) {
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

function buildBalancedRecords(items, matchers, fallbackPredicate, limit = 5) {
  const records = [];
  const seen = new Set();
  const add = (label, place) => {
    if (!place) return;
    const key = place.place_id || `${place.name}-${place.lat}-${place.lng}`;
    if (!key || seen.has(key)) return;
    seen.add(key);
    records.push(placeRecord(label, place));
  };

  for (const matcher of matchers) {
    add(matcher.label, items.find(matcher.predicate));
  }

  for (const place of uniqueTop(items, fallbackPredicate, limit)) {
    const label =
      typeof fallbackPredicate?.labelFor === "function"
        ? fallbackPredicate.labelFor(place)
        : matchers.find((matcher) => matcher.predicate(place))?.label ||
          "Nearby place";
    add(label, place);
    if (records.length >= limit) break;
  }

  return records.slice(0, limit);
}

function surroundingCounts(nearby) {
  const surroundings =
    nearby?.surroundings && typeof nearby.surroundings === "object"
      ? nearby.surroundings
      : {};
  return {
    transportation: Array.isArray(surroundings.transportation)
      ? surroundings.transportation.length
      : 0,
    healthcare: Array.isArray(surroundings.healthcare)
      ? surroundings.healthcare.length
      : 0,
    education: Array.isArray(surroundings.education)
      ? surroundings.education.length
      : 0,
    shopping: Array.isArray(surroundings.shopping)
      ? surroundings.shopping.length
      : 0,
  };
}

function hasMinimumSurroundings(nearby, minPerGroup = 3) {
  return Object.values(surroundingCounts(nearby)).every(
    (count) => count >= minPerGroup,
  );
}

function mergeRecords(primary, secondary, limit = 6) {
  const seen = new Set();
  const merged = [];

  for (const item of [...(primary || []), ...(secondary || [])]) {
    const key = item?.place_id || `${item?.label}-${item?.name}-${item?.distance_m}`;
    if (!key || seen.has(key)) continue;
    seen.add(key);
    merged.push(item);
    if (merged.length >= limit) break;
  }

  return merged;
}

function mergeNearbyResults(base, next) {
  if (!base) return next;
  if (!next) return base;

  return {
    ...base,
    surroundings: {
      transportation: mergeRecords(
        base.surroundings?.transportation,
        next.surroundings?.transportation,
        6,
      ),
      healthcare: mergeRecords(
        base.surroundings?.healthcare,
        next.surroundings?.healthcare,
        5,
      ),
      education: mergeRecords(
        base.surroundings?.education,
        next.surroundings?.education,
        6,
      ),
      shopping: mergeRecords(
        base.surroundings?.shopping,
        next.surroundings?.shopping,
        5,
      ),
    },
    transport: mergeTop([base.transport, next.transport], 8),
    health: mergeTop([base.health, next.health], 8),
    education: mergeTop([base.education, next.education], 8),
    shopping: mergeTop([base.shopping, next.shopping], 8),
    computed_at: new Date().toISOString(),
  };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function transportationLabel(place) {
  if (hasType(place, "amenity:charging_station")) return "EV Charging Station";
  if (
    hasType(place, "station:subway") ||
    hasType(place, "railway:subway_entrance") ||
    place?.tags?.subway === "yes"
  ) {
    return "Metro Station";
  }
  if (hasAnyType(place, ["railway:tram_stop", "station:light_rail"])) {
    return "Tram Station";
  }
  if (hasAnyType(place, ["highway:bus_stop", "amenity:bus_station"])) {
    return "Bus Stop";
  }
  if (hasType(place, "public_transport:station")) return "Transit Station";
  return "Transportation";
}

function healthcareLabel(place) {
  if (hasType(place, "amenity:hospital")) return "Hospital";
  if (hasType(place, "amenity:clinic")) return "Clinic";
  if (hasType(place, "amenity:doctors") || place?.tags?.healthcare === "doctor") {
    return "Family Doctor";
  }
  if (hasType(place, "amenity:dentist") || place?.tags?.healthcare === "dentist") {
    return "Dentist";
  }
  if (hasType(place, "amenity:pharmacy") || place?.tags?.healthcare === "pharmacy") {
    return "Pharmacy";
  }
  return "Healthcare";
}

function educationLabel(place) {
  if (hasAnyType(place, ["amenity:kindergarten", "amenity:childcare"])) {
    return "Kindergarten / Preschool";
  }
  if (hasAnyType(place, ["amenity:university", "amenity:college"])) {
    return "University";
  }
  const level = String(place?.tags?.["isced:level"] || "");
  if (level.startsWith("1") || nameMatches(place, /primary|grundschule/i)) {
    return "Primary School";
  }
  if (
    level.startsWith("2") ||
    nameMatches(place, /middle|secondary|mittelschule|realschule/i)
  ) {
    return "Middle School";
  }
  if (level.startsWith("3") || nameMatches(place, /high school|gymnasium|lyceum/i)) {
    return "High School";
  }
  if (hasType(place, "amenity:school")) return "School";
  return "Education";
}

function shoppingLabel(place) {
  if (hasAnyType(place, ["shop:mall", "building:retail"])) return "Shopping Mall";
  if (hasType(place, "shop:supermarket")) return "Supermarket";
  if (hasType(place, "shop:convenience")) return "Convenience Store";
  if (hasType(place, "shop:department_store")) return "Department Store";
  if (hasType(place, "shop:bakery")) return "Bakery";
  if (hasType(place, "shop:greengrocer")) return "Greengrocer";
  return "Shopping";
}

async function buildNearbyPlacesFromOsm({ lat, lng }) {
  const transportationFilters = [
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
  ];
  const healthcareFilters = [
    '["amenity"="hospital"]',
    '["amenity"="clinic"]',
    '["amenity"="doctors"]',
    '["amenity"="dentist"]',
    '["amenity"="pharmacy"]',
    '["healthcare"~"doctor|clinic|hospital|dentist|pharmacy"]',
  ];
  const educationFilters = [
    '["amenity"="kindergarten"]',
    '["amenity"="childcare"]',
    '["amenity"="school"]',
    '["amenity"="university"]',
    '["amenity"="college"]',
  ];
  const shoppingFilters = [
    '["shop"="mall"]',
    '["building"="retail"]',
    '["shop"="supermarket"]',
    '["shop"="convenience"]',
    '["shop"="department_store"]',
    '["shop"="bakery"]',
    '["shop"="greengrocer"]',
  ];

  const transportationItems = await osmNearbySearch({
    lat,
    lng,
    filters: transportationFilters,
    radiusMeters: 7000,
    limit: 500,
  });
  const healthcareItems = await osmNearbySearch({
    lat,
    lng,
    filters: healthcareFilters,
    radiusMeters: 7000,
    limit: 500,
  });
  const educationItems = await osmNearbySearch({
    lat,
    lng,
    filters: educationFilters,
    radiusMeters: 7000,
    limit: 500,
  });
  const shoppingItems = await osmNearbySearch({
    lat,
    lng,
    filters: shoppingFilters,
    radiusMeters: 7000,
    limit: 500,
  });
  const items = [
    ...transportationItems,
    ...healthcareItems,
    ...educationItems,
    ...shoppingItems,
  ].sort((a, b) => (a.distance_m || 0) - (b.distance_m || 0));

  const isTransportation = (item) =>
    hasAnyType(item, [
      "amenity:charging_station",
      "highway:bus_stop",
      "amenity:bus_station",
      "railway:tram_stop",
      "station:light_rail",
      "station:subway",
      "railway:subway_entrance",
      "public_transport:station",
    ]) || item?.tags?.subway === "yes";
  isTransportation.labelFor = transportationLabel;

  const isHealthcare = (item) =>
    hasAnyType(item, [
      "amenity:hospital",
      "amenity:clinic",
      "amenity:doctors",
      "amenity:dentist",
      "amenity:pharmacy",
    ]) ||
    ["doctor", "clinic", "hospital", "dentist", "pharmacy"].includes(
      String(item?.tags?.healthcare || ""),
    );
  isHealthcare.labelFor = healthcareLabel;

  const isEducation = (item) =>
    hasAnyType(item, [
      "amenity:kindergarten",
      "amenity:childcare",
      "amenity:school",
      "amenity:university",
      "amenity:college",
    ]);
  isEducation.labelFor = educationLabel;

  const isShopping = (item) =>
    hasAnyType(item, [
      "shop:mall",
      "building:retail",
      "shop:supermarket",
      "shop:convenience",
      "shop:department_store",
      "shop:bakery",
      "shop:greengrocer",
    ]);
  isShopping.labelFor = shoppingLabel;

  const transportation = buildBalancedRecords(
    transportationItems,
    [
      {
        label: "EV Charging Station",
        predicate: (item) => hasType(item, "amenity:charging_station"),
      },
      {
        label: "Metro Station",
        predicate: (item) =>
          hasType(item, "station:subway") ||
          hasType(item, "railway:subway_entrance") ||
          item?.tags?.subway === "yes",
      },
      {
        label: "Bus Stop",
        predicate: (item) =>
          hasAnyType(item, ["highway:bus_stop", "amenity:bus_station"]),
      },
      {
        label: "Tram Station",
        predicate: (item) =>
          hasAnyType(item, ["railway:tram_stop", "station:light_rail"]),
      },
    ],
    isTransportation,
    6,
  );

  const healthcare = buildBalancedRecords(
    healthcareItems,
    [
      {
        label: "Hospital",
        predicate: (item) => hasType(item, "amenity:hospital"),
      },
      {
        label: "Clinic",
        predicate: (item) => hasType(item, "amenity:clinic"),
      },
      {
        label: "Family Doctor",
        predicate: (item) =>
          hasType(item, "amenity:doctors") || item?.tags?.healthcare === "doctor",
      },
      {
        label: "Dentist",
        predicate: (item) =>
          hasType(item, "amenity:dentist") || item?.tags?.healthcare === "dentist",
      },
      {
        label: "Pharmacy",
        predicate: (item) =>
          hasType(item, "amenity:pharmacy") ||
          item?.tags?.healthcare === "pharmacy",
      },
    ],
    isHealthcare,
    5,
  );

  const education = buildBalancedRecords(
    educationItems,
    [
      {
        label: "Kindergarten / Preschool",
        predicate: (item) =>
          hasAnyType(item, ["amenity:kindergarten", "amenity:childcare"]),
      },
      {
        label: "Primary School",
        predicate: (item) =>
          hasType(item, "amenity:school") &&
          (String(item?.tags?.["isced:level"] || "").startsWith("1") ||
            nameMatches(item, /primary|grundschule/i)),
      },
      {
        label: "Middle School",
        predicate: (item) =>
          hasType(item, "amenity:school") &&
          (String(item?.tags?.["isced:level"] || "").startsWith("2") ||
            nameMatches(item, /middle|secondary|mittelschule|realschule/i)),
      },
      {
        label: "High School",
        predicate: (item) =>
          hasType(item, "amenity:school") &&
          (String(item?.tags?.["isced:level"] || "").startsWith("3") ||
            nameMatches(item, /high school|gymnasium|lyceum/i)),
      },
      {
        label: "University",
        predicate: (item) =>
          hasAnyType(item, ["amenity:university", "amenity:college"]),
      },
    ],
    isEducation,
    6,
  );

  const shopping = buildBalancedRecords(
    shoppingItems,
    [
      {
        label: "Shopping Mall",
        predicate: (item) => hasAnyType(item, ["shop:mall", "building:retail"]),
      },
      {
        label: "Supermarket",
        predicate: (item) => hasType(item, "shop:supermarket"),
      },
      {
        label: "Convenience Store",
        predicate: (item) => hasType(item, "shop:convenience"),
      },
      {
        label: "Department Store",
        predicate: (item) => hasType(item, "shop:department_store"),
      },
      {
        label: "Bakery",
        predicate: (item) => hasType(item, "shop:bakery"),
      },
    ],
    isShopping,
    5,
  );

  const nearest = (predicate) => items.find(predicate) || null;

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
      transportation,
      healthcare,
      education,
      shopping,
    },
    transport: [evCharging, metroStation, busStop, tramStation].flatMap((item) =>
      item?.place ? [item.place] : [],
    ),
    health: healthcare.flatMap((item) => (item?.place ? [item.place] : [])),
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
    if (process.env.ENABLE_GOOGLE_PLACES_LOOKUP !== "true") {
      let best = null;
      for (let attempt = 0; attempt < 3; attempt++) {
        const next = await buildNearbyPlacesFromOsm({ lat, lng });
        best = mergeNearbyResults(best, next);
        if (hasMinimumSurroundings(best, 3)) return best;
        if (attempt < 2) await sleep(800);
      }
      return best;
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
