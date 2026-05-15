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

  // NOTE: UI spec request: show Hospitals / Shopping / Education / Transportation (English)
  const hospitalPlaces = Array.isArray(nearby?.health) ? nearby.health : [];
  const educationPlaces = Array.isArray(nearby?.education)
    ? nearby.education
    : [];
  const shoppingPlaces = Array.isArray(nearby?.shopping) ? nearby.shopping : [];
  const transportPlaces = Array.isArray(nearby?.transport)
    ? nearby.transport
    : [];

  const hasNearbyData =
    hospitalPlaces.length > 0 ||
    educationPlaces.length > 0 ||
    shoppingPlaces.length > 0 ||
    transportPlaces.length > 0;

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
            Nearby places
          </h2>
          {nearby?.computed_at ? (
            <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono">
              Updated: {String(nearby.computed_at)}
            </div>
          ) : null}
        </div>

        {!hasNearbyData ? (
          <div className="mt-3 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            Nearby places are not available yet. They are computed after a valid
            address is saved.
          </div>
        ) : null}

        {/* UX change: categories stacked vertically (not side-by-side) */}
        <div className="mt-5 space-y-6">
          <div>
            <div className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 font-jetbrains-mono">
              Hospitals
            </div>
            <NearbyList
              places={hospitalPlaces}
              emptyText="No hospitals found."
            />
          </div>

          <div>
            <div className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 font-jetbrains-mono">
              Shopping
            </div>
            <NearbyList places={shoppingPlaces} emptyText="No places found." />
          </div>

          <div>
            <div className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 font-jetbrains-mono">
              Education
            </div>
            <NearbyList
              places={educationPlaces}
              emptyText="No schools/universities found."
            />
          </div>

          <div>
            <div className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 font-jetbrains-mono">
              Transportation
            </div>
            <NearbyList
              places={transportPlaces}
              emptyText="No stations found."
            />
          </div>
        </div>
      </div>
    </div>
  );
}
