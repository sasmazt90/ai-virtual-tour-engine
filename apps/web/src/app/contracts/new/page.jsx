import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import useUser from "@/utils/useUser";
import { Header } from "../../../components/Header";
import { ArrowLeft, FileText, Loader2 } from "lucide-react";
import { TEMPLATE_SCHEMA, FIELD_META } from "@/utils/contractSchema";

const TEMPLATE_OPTIONS = [
  { value: "rental_agreement", label: "Residential Lease Agreement (1 page)" },
  { value: "sale_agreement", label: "Purchase Agreement (10 pages)" },
];

const TEMPLATE_PREVIEW = {
  rental_agreement:
    "https://esign.com/wp-content/uploads/Simple-1-Page-Residential-Lease-Agreement.png",
  sale_agreement:
    "https://ucarecdn.com/2997fa8e-17e4-403b-ba19-2360d75ec944/-/format/auto/",
};

function buildPropertyAddress(p) {
  if (!p) return "";
  const parts = [p.address_line, p.city, p.postal_code, p.country].filter(
    Boolean,
  );
  return parts.join(", ");
}

function FieldRow({ fieldKey, value, required, onChange }) {
  const meta = FIELD_META[fieldKey] || { label: fieldKey, type: "text" };
  const isTextarea = meta.type === "textarea";
  const labelText = meta.label;

  const wrapperClassName =
    fieldKey === "PROPERTY_ADDRESS" ||
    fieldKey === "OWNER_ADDRESS" ||
    fieldKey === "CUSTOMER_ADDRESS"
      ? "space-y-2 sm:col-span-2"
      : "space-y-2";

  const requiredMark = required ? (
    <span className="text-red-600">*</span>
  ) : null;

  return (
    <div className={wrapperClassName}>
      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
        {labelText} {requiredMark}
      </label>
      {isTextarea ? (
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          rows={5}
          className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
          placeholder={meta.placeholder || ""}
        />
      ) : (
        <input
          type={meta.type || "text"}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
          placeholder={meta.placeholder || ""}
        />
      )}
    </div>
  );
}

export default function NewContractPage() {
  const { data: user, loading: userLoading } = useUser();

  const [templateType, setTemplateType] = useState(TEMPLATE_OPTIONS[0].value);
  const [clientId, setClientId] = useState("");
  const [propertyId, setPropertyId] = useState("");

  const [fields, setFields] = useState({});
  const [error, setError] = useState(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    const cid = params.get("clientId");
    const t = params.get("templateType");

    if (cid) setClientId(cid);

    const isAllowedTemplate = TEMPLATE_OPTIONS.some((o) => o.value === t);
    if (t && isAllowedTemplate) setTemplateType(t);
  }, []);

  const schemaKey = useMemo(() => {
    const t = String(templateType || "");
    if (t === "sale_agreement") return "sale_agreement";
    if (t === "rental_agreement") return "rental_agreement";
    return "legacy";
  }, [templateType]);

  const schema = TEMPLATE_SCHEMA[schemaKey] || TEMPLATE_SCHEMA.legacy;
  const requiredSet = useMemo(
    () => new Set(schema.required || []),
    [schema.required],
  );

  // Ensure fields object has keys for the selected template
  useEffect(() => {
    setFields((prev) => {
      const next = { ...prev };
      for (const sec of schema.sections) {
        for (const k of sec.fields) {
          if (next[k] === undefined || next[k] === null) next[k] = "";
        }
      }
      return next;
    });
  }, [schemaKey]);

  const { data: clients = [] } = useQuery({
    queryKey: ["clients", user?.id, "contracts-new-v2"],
    queryFn: async () => {
      const res = await fetch("/api/clients");
      if (!res.ok) throw new Error("Failed to load clients");
      return res.json();
    },
    enabled: !!user?.id,
  });

  const selectedClient = useMemo(() => {
    return clients.find((c) => c.id === clientId) || null;
  }, [clients, clientId]);

  // Pull interested properties for client
  const { data: interestedProperties = [] } = useQuery({
    queryKey: ["client-interested-properties", user?.id, clientId],
    queryFn: async () => {
      const res = await fetch(`/api/clients/${clientId}/interested-properties`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to load linked properties");
      }
      return res.json();
    },
    enabled: !!user?.id && !!clientId,
  });

  // Auto-pick most recent property
  useEffect(() => {
    if (!clientId) {
      setPropertyId("");
      return;
    }

    if (!Array.isArray(interestedProperties)) return;

    const first = interestedProperties[0];
    const nextId = first?.id || "";
    if (nextId && nextId !== propertyId) {
      setPropertyId(nextId);
    }
  }, [clientId, interestedProperties, propertyId]);

  const { data: propertyDetail } = useQuery({
    queryKey: ["property", user?.id, propertyId, "contracts-new-v2"],
    queryFn: async () => {
      const res = await fetch(`/api/properties/${propertyId}`);
      if (!res.ok) throw new Error("Failed to load property");
      return res.json();
    },
    enabled: !!user?.id && !!propertyId,
  });

  // Prefill from property + client (manual override always allowed)
  useEffect(() => {
    if (!propertyDetail) return;

    setFields((prev) => {
      const next = { ...prev };

      if (!next.PROPERTY_ADDRESS)
        next.PROPERTY_ADDRESS = buildPropertyAddress(propertyDetail);
      if (!next.PROPERTY_TYPE && propertyDetail.housing_type)
        next.PROPERTY_TYPE = String(propertyDetail.housing_type);

      if (
        !next.PROPERTY_SIZE &&
        propertyDetail.size_sqm !== null &&
        propertyDetail.size_sqm !== undefined
      ) {
        next.PROPERTY_SIZE = String(propertyDetail.size_sqm);
      }

      if (
        !next.ROOM_COUNT &&
        propertyDetail.rooms !== null &&
        propertyDetail.rooms !== undefined
      ) {
        next.ROOM_COUNT = String(propertyDetail.rooms);
      }

      if (
        !next.SALE_PRICE &&
        propertyDetail.price !== null &&
        propertyDetail.price !== undefined
      ) {
        next.SALE_PRICE = String(propertyDetail.price);
      }
      if (
        !next.RENT_PRICE &&
        propertyDetail.price !== null &&
        propertyDetail.price !== undefined
      ) {
        next.RENT_PRICE = String(propertyDetail.price);
      }
      if (!next.CURRENCY && propertyDetail.currency)
        next.CURRENCY = String(propertyDetail.currency);

      if (!next.OWNER_NAME && propertyDetail.owner_name)
        next.OWNER_NAME = String(propertyDetail.owner_name);
      if (!next.AGENT_NAME && user?.name) next.AGENT_NAME = String(user.name);

      return next;
    });
  }, [propertyDetail, user?.name]);

  useEffect(() => {
    if (!selectedClient) return;

    setFields((prev) => {
      const next = { ...prev };
      if (!next.CUSTOMER_NAME)
        next.CUSTOMER_NAME = selectedClient.full_name || "";
      if (!next.CUSTOMER_ADDRESS) {
        const addr = [selectedClient.city, selectedClient.country]
          .filter(Boolean)
          .join(", ");
        next.CUSTOMER_ADDRESS = addr;
      }
      return next;
    });
  }, [selectedClient]);

  const sortedClients = useMemo(() => {
    const list = clients.slice();
    list.sort((a, b) =>
      String(a.full_name || "").localeCompare(String(b.full_name || "")),
    );
    return list;
  }, [clients]);

  const createMutation = useMutation({
    mutationFn: async () => {
      setError(null);

      if (!clientId || !templateType) {
        throw new Error("Please select a template and a client");
      }

      const res = await fetch("/api/contracts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          clientId,
          templateType,
          // Optional: let backend auto-derive property if we didn't resolve one yet
          propertyId: propertyId || null,
          filledFields: fields,
          generatePdf: false,
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to create contract");
      }

      return res.json();
    },
    onSuccess: (contract) => {
      if (typeof window !== "undefined") {
        window.location.href = `/contracts/${contract.id}`;
      }
    },
    onError: (err) => {
      console.error(err);
      setError(err?.message || "Failed to create contract");
    },
  });

  if (userLoading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E]">
        <Header />
        <div className="pt-16 flex items-center justify-center min-h-screen">
          <p className="text-gray-600 dark:text-gray-400 font-jetbrains-mono">
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

  const hasManyProperties =
    Array.isArray(interestedProperties) && interestedProperties.length > 1;

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E]">
      <Header />

      <div className="pt-16">
        <div className="max-w-4xl mx-auto px-4 sm:px-8 py-8 sm:py-12">
          <a
            href={clientId ? `/directory/${clientId}` : "/contracts"}
            className="inline-flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 font-jetbrains-mono"
          >
            <ArrowLeft size={16} />
            Back
          </a>

          <div className="mt-4 mb-8">
            <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
              New Contract
            </h1>
            <p className="mt-2 text-gray-600 dark:text-gray-300 font-jetbrains-mono">
              Choose a template + client. We auto-fill the property from the
              client’s interested properties.
            </p>
          </div>

          <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6 sm:p-8">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              <div className="space-y-2 sm:col-span-2">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                  Contract Template
                </label>
                <select
                  value={templateType}
                  onChange={(e) => setTemplateType(e.target.value)}
                  className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                >
                  {TEMPLATE_OPTIONS.map((t) => (
                    <option key={t.value} value={t.value}>
                      {t.label}
                    </option>
                  ))}
                </select>

                {TEMPLATE_PREVIEW[templateType] ? (
                  <div className="mt-3 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-[#1E1E1E] overflow-hidden">
                    <img
                      src={TEMPLATE_PREVIEW[templateType]}
                      alt="Template preview"
                      className="w-full h-[240px] object-cover"
                    />
                  </div>
                ) : null}
              </div>

              <div className="space-y-2 sm:col-span-2">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                  Client
                </label>
                <select
                  value={clientId}
                  onChange={(e) => {
                    setClientId(e.target.value);
                    setPropertyId("");
                  }}
                  className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                >
                  <option value="">Select client...</option>
                  {sortedClients.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.full_name}
                    </option>
                  ))}
                </select>
                <p className="text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
                  Property will be auto-selected from this client’s interested
                  properties.
                </p>
              </div>

              {clientId ? (
                <div className="sm:col-span-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/40 p-4">
                  <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                    Linked property
                  </div>

                  {Array.isArray(interestedProperties) &&
                  interestedProperties.length === 0 ? (
                    <div className="mt-2 text-sm text-amber-700 dark:text-amber-300 font-jetbrains-mono">
                      This client isn’t linked to any property yet.
                    </div>
                  ) : null}

                  {hasManyProperties ? (
                    <div className="mt-3 space-y-2">
                      <label className="block text-xs text-gray-600 dark:text-gray-400 font-jetbrains-mono">
                        Choose which property this contract is for
                      </label>
                      <select
                        value={propertyId}
                        onChange={(e) => setPropertyId(e.target.value)}
                        className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
                      >
                        {interestedProperties.map((p) => (
                          <option key={p.id} value={p.id}>
                            {p.title}
                          </option>
                        ))}
                      </select>
                    </div>
                  ) : null}

                  {propertyDetail ? (
                    <div className="mt-3 text-sm text-gray-700 dark:text-gray-300 font-jetbrains-mono">
                      <div className="font-medium">{propertyDetail.title}</div>
                      <div className="text-xs text-gray-600 dark:text-gray-400">
                        {buildPropertyAddress(propertyDetail)}
                      </div>
                    </div>
                  ) : null}
                </div>
              ) : null}

              {/* Template-driven sections (optional prefill before opening contract detail) */}
              {schema.sections.map((sec) => (
                <div key={sec.title} className="sm:col-span-2">
                  <div className="mt-2 mb-3 flex items-center justify-between">
                    <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
                      {sec.title}
                    </h2>
                    <div className="text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
                      <span className="text-red-600">*</span> required
                    </div>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                    {sec.fields.map((key) => {
                      const required = requiredSet.has(key);
                      const value = fields?.[key] || "";
                      return (
                        <FieldRow
                          key={key}
                          fieldKey={key}
                          value={value}
                          required={required}
                          onChange={(val) =>
                            setFields((prev) => ({
                              ...prev,
                              [key]: val,
                            }))
                          }
                        />
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>

            {error ? (
              <div className="mt-6 rounded-lg bg-red-50 dark:bg-red-900/30 p-3 text-sm text-red-600 dark:text-red-400 font-jetbrains-mono">
                {error}
              </div>
            ) : null}

            <div className="mt-6 flex items-center justify-end">
              <button
                onClick={() => createMutation.mutate()}
                disabled={createMutation.isPending}
                className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-[var(--brand)] hover:bg-[var(--brandHover)] text-white rounded-lg font-medium transition-colors disabled:opacity-50 font-jetbrains-mono"
              >
                {createMutation.isPending ? (
                  <Loader2 size={18} className="animate-spin" />
                ) : (
                  <FileText size={18} />
                )}
                Create Draft
              </button>
            </div>

            <p className="mt-4 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
              After creating, you can manually fill fields and then click
              “Generate PDF” inside the contract.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
