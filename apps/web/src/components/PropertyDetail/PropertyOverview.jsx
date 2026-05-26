import { formatMoney } from "@/utils/formatters";

function formatMoneyWithCurrency(value, currency) {
  if (value === null || value === undefined) return "—";
  const base = formatMoney(value);
  const cur = currency ? String(currency).trim() : "";
  return cur ? `${base} ${cur}` : base;
}

function formatDistance(meters) {
  const m = Number(meters);
  if (!Number.isFinite(m)) return "";
  if (m >= 1000) {
    const km = Math.round((m / 1000) * 10) / 10;
    return `${km} km`;
  }
  return `${Math.round(m)} m`;
}

function parseNearbyPlaces(rawNearby) {
  if (!rawNearby) return null;

  // Depending on the Postgres driver/settings, jsonb can come back as an object or a string.
  if (typeof rawNearby === "string") {
    try {
      const parsed = JSON.parse(rawNearby);
      return parsed && typeof parsed === "object" ? parsed : null;
    } catch (error) {
      console.warn("parseNearbyPlaces: failed to parse nearby_places", error);
      return null;
    }
  }

  if (typeof rawNearby === "object") return rawNearby;
  return null;
}

function NearbyList({ places, emptyText }) {
  const list = Array.isArray(places) ? places : [];
  if (list.length === 0) {
    return (
      <div className="mt-2 text-sm text-gray-500 dark:text-gray-400 font-jetbrains-mono">
        {emptyText}
      </div>
    );
  }

  return (
    <div className="mt-2 space-y-2">
      {list.slice(0, 10).map((p, idx) => {
        const name = p?.name ? String(p.name) : "—";
        const dist = formatDistance(p?.distance_m);
        const key = p?.place_id || `${name}-${idx}`;

        return (
          <div
            key={key}
            className="flex items-center justify-between gap-4 text-sm text-gray-800 dark:text-gray-200 font-jetbrains-mono"
          >
            <div className="min-w-0 truncate">{name}</div>
            <div className="shrink-0 text-gray-500 dark:text-gray-400">
              {dist || ""}
            </div>
          </div>
        );
      })}
    </div>
  );
}

const SURROUNDING_GROUPS = [
  { key: "transportation", title: "Transportation" },
  { key: "healthcare", title: "Healthcare" },
  { key: "education", title: "Education" },
  { key: "shopping", title: "Shopping" },
];

function normalizeSurroundingItem(raw) {
  if (!raw || typeof raw !== "object") return null;

  const place = raw.place && typeof raw.place === "object" ? raw.place : raw;
  const label = raw.label ? String(raw.label) : "";
  const tags = place?.tags && typeof place.tags === "object" ? place.tags : {};
  const rawName = place?.name ? String(place.name) : "";
  const technicalNames = new Set([
    "subway_entrance",
    "bus_stop",
    "tram_stop",
    "station",
    "halt",
    "stop",
  ]);
  const name = technicalNames.has(rawName.toLowerCase())
    ? String(tags.description || tags.name || tags.ref || rawName)
    : rawName;
  const distance = raw.distance_m ?? place?.distance_m ?? null;

  if (!label && !name && distance === null) return null;

  return {
    label: label || name || "Nearby place",
    name,
    distance_m: distance,
    place_id: raw.place_id || place?.place_id || `${label}-${name}`,
  };
}

function groupSurroundingItems(items) {
  const groups = [];
  const byLabel = new Map();

  for (const item of Array.isArray(items) ? items : []) {
    const label = item?.label || "Nearby place";
    if (!byLabel.has(label)) {
      const group = { label, places: [] };
      byLabel.set(label, group);
      groups.push(group);
    }
    byLabel.get(label).places.push(item);
  }

  return groups.map((group) => ({
    ...group,
    places: group.places
      .slice()
      .sort((a, b) => Number(a?.distance_m || 0) - Number(b?.distance_m || 0)),
  }));
}

function buildSurroundingGroups(nearby) {
  const detailed =
    nearby?.surroundings && typeof nearby.surroundings === "object"
      ? nearby.surroundings
      : null;

  if (detailed) {
    return SURROUNDING_GROUPS.map((group) => ({
      ...group,
      items: (Array.isArray(detailed[group.key]) ? detailed[group.key] : [])
        .map(normalizeSurroundingItem)
        .filter(Boolean),
    }));
  }

  return [
    {
      key: "transportation",
      title: "Transportation",
      items: (Array.isArray(nearby?.transport) ? nearby.transport : [])
        .map((place) =>
          normalizeSurroundingItem({ label: "Nearby transport", place }),
        )
        .filter(Boolean),
    },
    {
      key: "healthcare",
      title: "Healthcare",
      items: (Array.isArray(nearby?.health) ? nearby.health : [])
        .map((place) => normalizeSurroundingItem({ label: "Healthcare", place }))
        .filter(Boolean),
    },
    {
      key: "education",
      title: "Education",
      items: (Array.isArray(nearby?.education) ? nearby.education : [])
        .map((place) => normalizeSurroundingItem({ label: "Education", place }))
        .filter(Boolean),
    },
    {
      key: "shopping",
      title: "Shopping",
      items: (Array.isArray(nearby?.shopping) ? nearby.shopping : [])
        .map((place) => normalizeSurroundingItem({ label: "Shopping", place }))
        .filter(Boolean),
    },
  ];
}

function SurroundingList({ items }) {
  const list = Array.isArray(items) ? items : [];
  if (list.length === 0) {
    return (
      <div className="mt-2 text-sm text-gray-500 dark:text-gray-400 font-jetbrains-mono">
        No nearby places found.
      </div>
    );
  }

  const grouped = groupSurroundingItems(list).slice(0, 10);

  return (
    <div className="mt-3 space-y-3">
      {grouped.map((group) => {
        const nearest = group.places[0];
        const key = `${group.label}-${nearest?.place_id || nearest?.name || ""}`;

        return (
          <div
            key={key}
            className="rounded-lg border border-gray-200/70 bg-gray-50/70 p-3 dark:border-gray-700/70 dark:bg-[#202020]"
          >
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <div className="font-medium text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                  {group.label}
                </div>
                <div className="mt-0.5 text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
                  {group.places.length} nearby{" "}
                  {group.places.length === 1 ? "place" : "places"}
                </div>
              </div>
            </div>

            <div className="mt-3 space-y-1.5">
              {group.places.slice(0, 5).map((p, idx) => {
                const dist = formatDistance(p?.distance_m);
                const placeKey = p?.place_id || `${group.label}-${p?.name}-${idx}`;
                return (
                  <div
                    key={placeKey}
                    className="grid grid-cols-[minmax(0,1fr)_auto] gap-4 text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono"
                  >
                    <div className="min-w-0 truncate">
                      {p?.name || group.label}
                    </div>
                    <div className="shrink-0 text-gray-500 dark:text-gray-400">
                      {dist || ""}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Field({ label, value }) {
  return (
    <div className="text-gray-600 dark:text-gray-300 font-jetbrains-mono">
      <div className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">
        {label}
      </div>
      <div className="mt-1 font-medium text-gray-900 dark:text-gray-100">
        {value}
      </div>
    </div>
  );
}

function ChipList({ items }) {
  const list = Array.isArray(items) ? items : [];
  if (list.length === 0) {
    return <div className="text-gray-900 dark:text-gray-100">—</div>;
  }

  return (
    <div className="mt-2 flex flex-wrap gap-2">
      {list.map((raw) => {
        const label = raw === null || raw === undefined ? "" : String(raw);
        if (!label.trim()) return null;
        return (
          <div
            key={label}
            className="px-2.5 py-1 rounded-full text-xs border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 text-gray-800 dark:text-gray-200"
          >
            {label}
          </div>
        );
      })}
    </div>
  );
}

export function PropertyOverview({ property }) {
  const currency = property.currency ? String(property.currency) : "";
  const priceText = formatMoneyWithCurrency(property.price, currency);

  const roomsText =
    property.bedrooms !== null && property.bedrooms !== undefined
      ? `${property.bedrooms || 0} + ${property.living_rooms || 0}`
      : (property.rooms ?? "—");

  const interior = Array.isArray(property.features_interior)
    ? property.features_interior
    : [];
  const exterior = Array.isArray(property.features_exterior)
    ? property.features_exterior
    : [];

  const nearby = parseNearbyPlaces(property?.nearby_places);
  const surroundingGroups = buildSurroundingGroups(nearby);
  const hasNearbyData = surroundingGroups.some(
    (group) => group.items.length > 0,
  );

  return (
    <div className="space-y-8">
      <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 font-jetbrains-mono">
          Overview
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
          <Field
            label="Status"
            value={
              property.property_status === "for_sale" ? "For Sale" : "For Rent"
            }
          />
          <Field label="Price" value={priceText} />

          {/* Translate remaining Turkish UI labels to English */}
          <Field label="Housing type" value={property.housing_type || "—"} />
          <Field label="Layout" value={property.housing_shape || "—"} />

          <Field label="Rooms" value={roomsText || "—"} />
          <Field label="Bathrooms" value={property.bathrooms ?? "—"} />

          <Field
            label="Gross / Net"
            value={
              property.gross_area_sqm || property.net_area_sqm
                ? `${property.gross_area_sqm ?? "—"} / ${property.net_area_sqm ?? "—"}`
                : "—"
            }
          />
          <Field label="Total floors" value={property.total_floors ?? "—"} />

          <Field label="Floor number" value={property.floor_number ?? "—"} />
          <Field label="Building age" value={property.building_age ?? "—"} />

          <Field label="Heating" value={property.heating_type || "—"} />
          <Field
            label="Elevator"
            value={
              property.elevator === true
                ? "Yes"
                : property.elevator === false
                  ? "No"
                  : "—"
            }
          />

          <Field label="Parking" value={property.parking_type || "—"} />
          <Field label="Title deed" value={property.title_deed_status || "—"} />

          <Field label="Furnishing" value={property.furnished_status || "—"} />
          <Field
            label="Mortgage eligible"
            value={
              property.mortgage_eligible === true
                ? "Yes"
                : property.mortgage_eligible === false
                  ? "No"
                  : "—"
            }
          />

          <Field
            label="Construction type"
            value={property.construction_type || "—"}
          />
          <Field label="Usage status" value={property.usage_status || "—"} />

          <Field label="Facade" value={property.facade || "—"} />
          <Field
            label="Deposit"
            value={
              property.deposit !== null && property.deposit !== undefined
                ? formatMoney(property.deposit)
                : "—"
            }
          />

          <Field
            label="Dues"
            value={
              property.dues !== null && property.dues !== undefined
                ? formatMoney(property.dues)
                : "—"
            }
          />

          <div className="sm:col-span-2 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            <div className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">
              Description
            </div>
            <div className="mt-1 text-gray-900 dark:text-gray-100 whitespace-pre-wrap">
              {property.description || "—"}
            </div>
          </div>
        </div>
      </div>

      {interior.length > 0 || exterior.length > 0 ? (
        <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 font-jetbrains-mono">
            Features
          </h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 text-sm">
            <div className="text-gray-600 dark:text-gray-300 font-jetbrains-mono">
              <div className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">
                Interior features
              </div>
              <ChipList items={interior} />
            </div>

            <div className="text-gray-600 dark:text-gray-300 font-jetbrains-mono">
              <div className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">
                Exterior features
              </div>
              <ChipList items={exterior} />
            </div>
          </div>
        </div>
      ) : null}

      <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6">
        <div className="flex items-center justify-between gap-4">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Surroundings
          </h2>
          {nearby?.computed_at ? (
            <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
              Updated: {String(nearby.computed_at)}
            </div>
          ) : null}
        </div>

        {!hasNearbyData ? (
          <div className="mt-3 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            Surroundings are not available yet. They are computed after a valid
            address is saved.
          </div>
        ) : null}

        <div className="mt-5 space-y-6">
          {surroundingGroups.map((group) => (
            <div key={group.key}>
              <div className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 font-jetbrains-mono">
                {group.title}
              </div>
              <SurroundingList items={group.items} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
