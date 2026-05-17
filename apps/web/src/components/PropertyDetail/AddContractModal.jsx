import { useCallback, useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FileUp, FileText, Loader2 } from "lucide-react";
import useUpload from "@/utils/useUpload";
import { ModalShell } from "./ModalShell";
import { ClientCombobox } from "@/components/Calendar/ClientCombobox";

const TEMPLATE_OPTIONS = [
  { value: "agency_authorization", label: "Agency Authorization" },
  { value: "buyer_representation", label: "Buyer Representation Agreement" },
  { value: "handover_protocol", label: "Property Handover Protocol" },
  { value: "offer_letter", label: "Property Offer Letter" },
  { value: "rental_agreement", label: "Residential Lease Agreement" },
  { value: "sale_agreement", label: "Residential Purchase Agreement" },
  { value: "seller_listing_agreement", label: "Seller Listing Agreement" },
  { value: "tenant_representation", label: "Tenant Representation Agreement" },
  { value: "viewing_report", label: "Viewing and Inspection Report" },
];

const FIELD_LABELS = {
  OWNER_NAME: "Owner name",
  OWNER_EMAIL: "Owner email",
  OWNER_PHONE: "Owner phone",
  OWNER_ADDRESS: "Owner address",
  CUSTOMER_NAME: "Client name",
  CUSTOMER_EMAIL: "Client email",
  CUSTOMER_PHONE: "Client phone",
  CUSTOMER_ADDRESS: "Client address",
  PROPERTY_TITLE: "Property title",
  PROPERTY_ADDRESS: "Property address",
  PROPERTY_TYPE: "Property type",
  PROPERTY_SIZE: "Size",
  ROOM_COUNT: "Rooms",
  FLOOR_INFO: "Floor",
  TITLE_DEED_INFO: "Title deed",
  SALE_PRICE: "Sale price",
  RENT_PRICE: "Rent",
  CURRENCY: "Currency",
  DEPOSIT_AMOUNT: "Deposit",
  RENT_START_DATE: "Start date",
  RENT_DURATION: "Duration",
  PAYMENT_METHOD: "Payment method",
  AGENCY_NAME: "Agency",
  AGENT_NAME: "Agent name",
  AGENT_CONTACT: "Agent contact",
  additionalTerms: "Special terms",
};

const COMMON_FIELDS = [
  "PROPERTY_TITLE",
  "PROPERTY_ADDRESS",
  "PROPERTY_TYPE",
  "PROPERTY_SIZE",
  "ROOM_COUNT",
  "FLOOR_INFO",
  "OWNER_NAME",
  "OWNER_EMAIL",
  "OWNER_PHONE",
  "CUSTOMER_NAME",
  "CUSTOMER_EMAIL",
  "CUSTOMER_PHONE",
  "AGENCY_NAME",
  "AGENT_NAME",
  "AGENT_CONTACT",
  "additionalTerms",
];

const TEMPLATE_FIELDS = {
  sale_agreement: [
    "SALE_PRICE",
    "CURRENCY",
    "PAYMENT_METHOD",
    "TITLE_DEED_INFO",
    ...COMMON_FIELDS,
  ],
  rental_agreement: [
    "RENT_PRICE",
    "CURRENCY",
    "DEPOSIT_AMOUNT",
    "RENT_START_DATE",
    "RENT_DURATION",
    ...COMMON_FIELDS,
  ],
  seller_listing_agreement: [
    "SALE_PRICE",
    "CURRENCY",
    "PAYMENT_METHOD",
    ...COMMON_FIELDS,
  ],
  buyer_representation: [
    "SALE_PRICE",
    "CURRENCY",
    "PAYMENT_METHOD",
    ...COMMON_FIELDS,
  ],
  tenant_representation: [
    "RENT_PRICE",
    "CURRENCY",
    "DEPOSIT_AMOUNT",
    "RENT_START_DATE",
    ...COMMON_FIELDS,
  ],
  offer_letter: [
    "SALE_PRICE",
    "CURRENCY",
    "PAYMENT_METHOD",
    "TITLE_DEED_INFO",
    ...COMMON_FIELDS,
  ],
  viewing_report: ["RENT_START_DATE", ...COMMON_FIELDS],
  handover_protocol: [
    "DEPOSIT_AMOUNT",
    "RENT_START_DATE",
    "TITLE_DEED_INFO",
    ...COMMON_FIELDS,
  ],
  agency_authorization: ["RENT_START_DATE", ...COMMON_FIELDS],
};

function isPdfMime(mimeType) {
  if (!mimeType) return false;
  const s = String(mimeType).toLowerCase();
  return s === "application/pdf" || s.endsWith("/pdf");
}

function joinAddress(parts) {
  return parts.filter(Boolean).join(", ");
}

function safeValue(value) {
  if (value === null || value === undefined) return "";
  return String(value);
}

function buildFloorInfo(property) {
  const floor = safeValue(property?.floor_number);
  const total = safeValue(property?.total_floors);
  if (floor && total) return `${floor} / ${total}`;
  return floor || total || "";
}

function buildContractFieldDefaults({ property, client }) {
  const today = new Date().toISOString().slice(0, 10);
  const propertyAddress = joinAddress([
    property?.address_line,
    property?.city,
    property?.postal_code,
    property?.country,
  ]);
  const clientAddress = joinAddress([client?.city, client?.country]);
  const ownerAddress = joinAddress([
    property?.owner_city,
    property?.owner_country,
  ]);

  return {
    OWNER_NAME: safeValue(property?.owner_name),
    OWNER_EMAIL: safeValue(property?.owner_email),
    OWNER_PHONE: safeValue(property?.owner_phone),
    OWNER_ADDRESS: ownerAddress,
    CUSTOMER_NAME: safeValue(client?.full_name),
    CUSTOMER_EMAIL: safeValue(client?.email),
    CUSTOMER_PHONE: safeValue(client?.phone),
    CUSTOMER_ADDRESS: clientAddress,
    PROPERTY_TITLE: safeValue(property?.title),
    PROPERTY_ADDRESS: propertyAddress,
    PROPERTY_TYPE: safeValue(property?.housing_type),
    PROPERTY_SIZE:
      safeValue(property?.gross_area_sqm) || safeValue(property?.size_sqm),
    ROOM_COUNT: safeValue(property?.rooms),
    FLOOR_INFO: buildFloorInfo(property),
    TITLE_DEED_INFO: safeValue(property?.title_deed_status),
    SALE_PRICE: safeValue(property?.price),
    RENT_PRICE: safeValue(property?.price),
    CURRENCY: safeValue(property?.currency),
    DEPOSIT_AMOUNT: safeValue(property?.deposit),
    RENT_START_DATE: today,
    RENT_DURATION: "",
    PAYMENT_METHOD: "",
    AGENCY_NAME: "",
    AGENT_NAME: "",
    AGENT_CONTACT: "",
    additionalTerms: "",
  };
}

function uniqueFields(templateType) {
  const fields = TEMPLATE_FIELDS[templateType] || COMMON_FIELDS;
  return Array.from(new Set(fields));
}

export default function AddContractModal({
  open,
  onClose,
  property,
  propertyId,
  userId,
  defaultClientId,
}) {
  const queryClient = useQueryClient();

  const [mode, setMode] = useState("upload");
  const [error, setError] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [templateType, setTemplateType] = useState(TEMPLATE_OPTIONS[0]?.value);
  const [clientId, setClientId] = useState(defaultClientId || "");
  const [fieldOverrides, setFieldOverrides] = useState({});

  const [upload, { loading: uploadLoading }] = useUpload();

  const { data: clients = [], isLoading: clientsLoading } = useQuery({
    queryKey: ["clients", userId],
    queryFn: async () => {
      const res = await fetch("/api/clients?type=all");
      if (!res.ok) throw new Error("Could not load clients");
      return res.json();
    },
    enabled: !!open && !!userId,
  });

  const selectedClient = useMemo(() => {
    return clients.find((c) => String(c.id) === String(clientId)) || null;
  }, [clientId, clients]);

  useEffect(() => {
    if (!open || mode !== "generate") return;
    setFieldOverrides(
      buildContractFieldDefaults({ property, client: selectedClient }),
    );
  }, [clientId, mode, open, property, selectedClient, templateType]);

  const resetState = useCallback(() => {
    setError(null);
    setMode("upload");
    setSelectedFile(null);
    setTemplateType(TEMPLATE_OPTIONS[0]?.value);
    setClientId(defaultClientId || "");
    setFieldOverrides({});
  }, [defaultClientId]);

  const closeAndReset = useCallback(() => {
    resetState();
    onClose();
  }, [onClose, resetState]);

  const invalidateProperty = useCallback(async () => {
    if (!userId || !propertyId) return;
    await queryClient.invalidateQueries({
      queryKey: ["property", userId, propertyId],
    });
  }, [propertyId, queryClient, userId]);

  const uploadMutation = useMutation({
    mutationFn: async () => {
      setError(null);
      if (!propertyId) throw new Error("Missing property context.");
      if (!selectedFile) throw new Error("Please select a PDF file.");

      const { url, mimeType, error: upErr } = await upload({
        file: selectedFile,
      });
      if (upErr) throw new Error(upErr);
      if (!isPdfMime(mimeType)) throw new Error("Please upload a PDF file.");

      const res = await fetch("/api/contracts/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          propertyId,
          pdfUrl: url,
          fileName: selectedFile?.name || null,
          clientId: clientId || null,
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not upload contract.");
      }

      return res.json();
    },
    onSuccess: async () => {
      await invalidateProperty();
      closeAndReset();
    },
    onError: (err) => {
      console.error(err);
      setError(err?.message || "Could not upload contract.");
    },
  });

  const generateMutation = useMutation({
    mutationFn: async () => {
      setError(null);
      if (!propertyId) throw new Error("Missing property context.");
      if (!templateType) throw new Error("Please select a template.");

      const res = await fetch("/api/contracts/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          propertyId,
          templateType,
          clientId: clientId || null,
          fieldOverrides,
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not generate contract.");
      }

      return res.json();
    },
    onSuccess: async () => {
      await invalidateProperty();
      closeAndReset();
    },
    onError: (err) => {
      console.error(err);
      setError(err?.message || "Could not generate contract.");
    },
  });

  const isBusy =
    uploadLoading || uploadMutation.isPending || generateMutation.isPending;

  const primaryLabel = useMemo(() => {
    if (mode === "upload") return "Upload PDF";
    return "Create Contract";
  }, [mode]);

  const onPrimary = useCallback(() => {
    if (mode === "upload") uploadMutation.mutate();
    else generateMutation.mutate();
  }, [generateMutation, mode, uploadMutation]);

  if (!open) return null;

  const visibleFields = uniqueFields(templateType);

  return (
    <ModalShell
      title="Add Contract"
      onClose={isBusy ? () => {} : closeAndReset}
    >
      <div className="space-y-4 font-jetbrains-mono">
        <div className="space-y-2">
          <div className="text-sm text-gray-700 dark:text-gray-300">
            Client
          </div>
          <ClientCombobox
            value={clientId}
            onChange={setClientId}
            clients={clients}
            placeholder={
              clientsLoading ? "Loading clients..." : "Select a client"
            }
          />
          <div className="text-xs text-gray-600 dark:text-gray-400">
            The contract will be saved on this property and linked to the
            selected client.
          </div>
        </div>

        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => {
              setError(null);
              setMode("upload");
            }}
            disabled={isBusy}
            className={`px-3 py-2 rounded-lg border text-sm ${
              mode === "upload"
                ? "bg-gray-900 text-white border-gray-900"
                : "bg-white dark:bg-[#262626] text-gray-900 dark:text-gray-100 border-gray-200 dark:border-gray-700"
            }`}
          >
            Upload PDF
          </button>

          <button
            type="button"
            onClick={() => {
              setError(null);
              setMode("generate");
            }}
            disabled={isBusy}
            className={`px-3 py-2 rounded-lg border text-sm ${
              mode === "generate"
                ? "bg-gray-900 text-white border-gray-900"
                : "bg-white dark:bg-[#262626] text-gray-900 dark:text-gray-100 border-gray-200 dark:border-gray-700"
            }`}
          >
            Use template
          </button>
        </div>

        {mode === "upload" ? (
          <div className="space-y-2">
            <div className="text-sm text-gray-700 dark:text-gray-300">
              Upload PDF
            </div>
            <input
              type="file"
              accept="application/pdf"
              disabled={isBusy}
              onChange={(e) => {
                const file = e.target.files?.[0] || null;
                setSelectedFile(file);
              }}
              className="block w-full text-sm text-gray-700 dark:text-gray-200"
            />
            <div className="text-xs text-gray-600 dark:text-gray-400">
              {selectedFile ? selectedFile.name : "PDF files only."}
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="space-y-2">
              <div className="text-sm text-gray-700 dark:text-gray-300">
                Template
              </div>
              <select
                value={templateType || ""}
                disabled={isBusy}
                onChange={(e) => setTemplateType(e.target.value)}
                className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)]"
              >
                {TEMPLATE_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
              <div className="text-xs text-gray-600 dark:text-gray-400">
                Review and edit the details before creating the PDF.
              </div>
            </div>

            <div className="rounded-lg border border-gray-200 dark:border-gray-700 p-3 bg-white dark:bg-gray-900">
              <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">
                Contract details
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {visibleFields.map((key) => {
                  const isLong = key === "additionalTerms";
                  const value = fieldOverrides?.[key] || "";
                  const label = FIELD_LABELS[key] || key;
                  return (
                    <label
                      key={key}
                      className={isLong ? "sm:col-span-2" : ""}
                    >
                      <span className="block text-xs text-gray-600 dark:text-gray-400 mb-1">
                        {label}
                      </span>
                      {isLong ? (
                        <textarea
                          rows={4}
                          value={value}
                          disabled={isBusy}
                          onChange={(e) =>
                            setFieldOverrides((prev) => ({
                              ...prev,
                              [key]: e.target.value,
                            }))
                          }
                          className="w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-sm text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)]"
                        />
                      ) : (
                        <input
                          value={value}
                          disabled={isBusy}
                          onChange={(e) =>
                            setFieldOverrides((prev) => ({
                              ...prev,
                              [key]: e.target.value,
                            }))
                          }
                          className="w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-sm text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)]"
                        />
                      )}
                    </label>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {isBusy ? (
          <div className="text-sm text-gray-600 dark:text-gray-300">
            {mode === "upload" ? "Uploading..." : "Creating contract..."}
          </div>
        ) : null}

        {error ? (
          <div className="rounded-lg bg-red-50 dark:bg-red-900/30 p-3 text-sm text-red-600 dark:text-red-400">
            {error}
          </div>
        ) : null}

        <div className="flex flex-col sm:flex-row gap-3">
          <button
            type="button"
            onClick={onPrimary}
            disabled={isBusy || !propertyId}
            className="inline-flex items-center justify-center gap-2 w-full px-6 py-3 bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 rounded-lg font-medium transition-colors hover:bg-gray-800 dark:hover:bg-gray-200 disabled:opacity-50"
          >
            {isBusy ? (
              <Loader2 size={18} className="animate-spin" />
            ) : mode === "upload" ? (
              <FileUp size={18} />
            ) : (
              <FileText size={18} />
            )}
            {primaryLabel}
          </button>

          <button
            type="button"
            onClick={closeAndReset}
            disabled={isBusy}
            className="inline-flex items-center justify-center gap-2 w-full px-6 py-3 rounded-lg font-medium transition-colors border border-gray-200 dark:border-gray-700 bg-white dark:bg-[#262626] text-gray-900 dark:text-gray-100 hover:bg-gray-50 dark:hover:bg-gray-800 disabled:opacity-50"
          >
            Cancel
          </button>
        </div>
      </div>
    </ModalShell>
  );
}
