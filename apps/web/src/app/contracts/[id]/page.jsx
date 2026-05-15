import { useEffect, useMemo, useState } from "react";
import useUser from "@/utils/useUser";
import { Header } from "@/components/Header";
import {
  TEMPLATE_SCHEMA,
  getEditableFieldSet,
} from "@/utils/contractSchema.js";
import { useContractDetail } from "@/hooks/useContractDetail";
import {
  getPdfMetadata,
  getSignatureMetadata,
  getAuditEntries,
} from "@/utils/contractHelpers";
import { ContractHeader } from "@/components/ContractDetail/ContractHeader";
import { ContractFieldsEditor } from "@/components/ContractDetail/ContractFieldsEditor";
import { SignatureSection } from "@/components/ContractDetail/SignatureSection";
import { AuditSection } from "@/components/ContractDetail/AuditSection";
import { FilledFieldsDebug } from "@/components/ContractDetail/FilledFieldsDebug";
import { PDFPreview } from "@/components/ContractDetail/PDFPreview";

function buildPropertyAddressFromContractRow(c) {
  const parts = [c?.address_line, c?.city, c?.postal_code, c?.country].filter(
    Boolean,
  );
  return parts.join(", ");
}

function buildClientAddressFromContractRow(city, country) {
  const parts = [city, country].filter(Boolean);
  return parts.join(", ");
}

function buildFloorInfoFromContractRow(c) {
  const floor =
    c?.floor_number === null || c?.floor_number === undefined
      ? ""
      : String(c.floor_number);
  const total =
    c?.total_floors === null || c?.total_floors === undefined
      ? ""
      : String(c.total_floors);
  if (floor && total) return `${floor} / ${total}`;
  return floor || total || "";
}

export default function ContractDetailPage(props) {
  const contractId = props?.params?.id;
  const { data: user, loading: userLoading } = useUser();

  const {
    contract,
    isLoading,
    error,
    markSignedMutation,
    markUnsignedMutation,
    regeneratePdfMutation,
    updateFieldsMutation,
  } = useContractDetail(contractId, user?.id);

  const [signedByAgentName, setSignedByAgentName] = useState("");
  const [signedByClientName, setSignedByClientName] = useState("");
  const [localError, setLocalError] = useState(null);
  const [editFields, setEditFields] = useState({});

  const schemaKey = useMemo(() => {
    const t = String(contract?.template_type || "");
    if (t === "sale_agreement") return "sale_agreement";
    if (t === "rental_agreement") return "rental_agreement";
    return "legacy";
  }, [contract?.template_type]);

  const schema = TEMPLATE_SCHEMA[schemaKey] || TEMPLATE_SCHEMA.legacy;
  const requiredSet = useMemo(
    () => new Set(schema.required || []),
    [schema.required],
  );

  const editableSet = useMemo(() => {
    return getEditableFieldSet(contract?.template_type);
  }, [contract?.template_type]);

  const suggestionsMap = useMemo(() => {
    if (!contract) return {};

    const propertyAddress = buildPropertyAddressFromContractRow(contract);
    const ownerAddress = buildClientAddressFromContractRow(
      contract?.owner_city,
      contract?.owner_country,
    );
    const customerAddress = buildClientAddressFromContractRow(
      contract?.client_city,
      contract?.client_country,
    );

    const floorInfo = buildFloorInfoFromContractRow(contract);

    const currency = contract?.currency ? String(contract.currency) : "";
    const priceRaw =
      contract?.price !== null && contract?.price !== undefined
        ? String(contract.price)
        : "";

    const agent =
      contract?.agent && typeof contract.agent === "object"
        ? contract.agent
        : {};

    const agentName = agent?.agent_name ? String(agent.agent_name) : "";
    const agencyName = agent?.company_name ? String(agent.company_name) : "";
    const agentEmail = agent?.agent_email ? String(agent.agent_email) : "";

    const housingType = contract?.housing_type
      ? String(contract.housing_type)
      : "";
    const furnished = contract?.furnished_status
      ? String(contract.furnished_status)
      : "";

    const enumOptions =
      contract?.enumOptions && typeof contract.enumOptions === "object"
        ? contract.enumOptions
        : {};

    const propertyTypeOptions = Array.isArray(enumOptions.PROPERTY_TYPE)
      ? enumOptions.PROPERTY_TYPE
      : [];

    const furnishedOptions = Array.isArray(enumOptions.FURNISHED_STATUS)
      ? enumOptions.FURNISHED_STATUS
      : [];

    const size =
      contract?.gross_area_sqm !== null &&
      contract?.gross_area_sqm !== undefined
        ? String(contract.gross_area_sqm)
        : contract?.size_sqm !== null && contract?.size_sqm !== undefined
          ? String(contract.size_sqm)
          : "";

    const rooms =
      contract?.rooms !== null && contract?.rooms !== undefined
        ? String(contract.rooms)
        : "";

    const deposit =
      contract?.deposit !== null && contract?.deposit !== undefined
        ? String(contract.deposit)
        : "";

    const propertyTypeSuggestions = [
      housingType,
      ...propertyTypeOptions,
    ].filter((v) => typeof v === "string" && v.trim().length > 0);

    const furnishedSuggestions = [furnished, ...furnishedOptions].filter(
      (v) => typeof v === "string" && v.trim().length > 0,
    );

    return {
      // PROPERTY_* suggestions
      PROPERTY_ADDRESS: [propertyAddress].filter(Boolean),
      PROPERTY_TYPE: propertyTypeSuggestions,
      PROPERTY_SIZE: [size].filter(Boolean),
      ROOM_COUNT: [rooms].filter(Boolean),
      FLOOR_INFO: [floorInfo].filter(Boolean),
      TITLE_DEED_INFO: [
        contract?.title_deed_status ? String(contract.title_deed_status) : "",
      ].filter(Boolean),
      FURNISHED_STATUS: furnishedSuggestions,

      // OWNER_* suggestions
      OWNER_NAME: [
        contract?.owner_name ? String(contract.owner_name) : "",
      ].filter(Boolean),
      OWNER_ADDRESS: [ownerAddress].filter(Boolean),

      // CUSTOMER_* suggestions
      CUSTOMER_NAME: [
        contract?.client_name ? String(contract.client_name) : "",
      ].filter(Boolean),
      CUSTOMER_ADDRESS: [customerAddress].filter(Boolean),

      // AGENT_* / AGENCY_*
      AGENCY_NAME: [agencyName].filter(Boolean),
      AGENT_NAME: [agentName].filter(Boolean),
      AGENT_CONTACT: [agentEmail].filter(Boolean),

      // Prices
      SALE_PRICE: [priceRaw].filter(Boolean),
      RENT_PRICE: [priceRaw].filter(Boolean),
      CURRENCY: [currency].filter(Boolean),
      DEPOSIT_AMOUNT: [deposit].filter(Boolean),

      // Legacy compatibility (still useful if you open an old contract)
      propertyAddress: [propertyAddress].filter(Boolean),
      price: [
        priceRaw && currency ? `${priceRaw} ${currency}`.trim() : priceRaw,
      ].filter(Boolean),
      clientName: [
        contract?.client_name ? String(contract.client_name) : "",
      ].filter(Boolean),
      clientEmail: [
        contract?.client_email ? String(contract.client_email) : "",
      ].filter(Boolean),
      clientPhone: [
        contract?.client_phone ? String(contract.client_phone) : "",
      ].filter(Boolean),
      agentName: [agentName].filter(Boolean),
      company: [agencyName].filter(Boolean),
    };
  }, [contract]);

  useEffect(() => {
    if (!contract) return;
    const f = contract.filled_fields || {};

    const nextAgent =
      f.signed_by_agent_name || f.agentName || user?.name || "" || "";
    const nextClient = f.signed_by_client_name || f.clientName || "";

    setSignedByAgentName(String(nextAgent || ""));
    setSignedByClientName(String(nextClient || ""));

    const nextEdit = {};
    for (const sec of schema.sections) {
      for (const key of sec.fields) {
        nextEdit[key] =
          f?.[key] !== undefined && f?.[key] !== null ? String(f[key]) : "";
      }
    }
    setEditFields(nextEdit);
  }, [contract, schema.sections, user?.name]);

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

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E]">
        <Header />
        <div className="pt-16 flex items-center justify-center min-h-screen">
          <p className="text-gray-600 dark:text-gray-400 font-jetbrains-mono">
            Loading contract...
          </p>
        </div>
      </div>
    );
  }

  if (error || !contract) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E]">
        <Header />
        <div className="pt-16 max-w-4xl mx-auto px-4 sm:px-8 py-12">
          <a
            href="/contracts"
            className="inline-flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 font-jetbrains-mono"
          >
            Back to Contracts
          </a>
          <div className="mt-6 rounded-xl bg-white dark:bg-[#262626] p-6 shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700">
            <p className="text-red-600 dark:text-red-400 font-jetbrains-mono">
              Could not load this contract.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const { pdfInlineUrl, pdfDownloadUrl, pdfMessage, pdfStatus } =
    getPdfMetadata(contract);
  const { signatureMethod, signedStatus, signedAt, isSigned } =
    getSignatureMetadata(contract);
  const auditEntries = getAuditEntries(contract);

  const propertyLink = contract.property_id
    ? `/properties/${contract.property_id}`
    : "/properties";

  const pdfRegenBlocked = isSigned;
  const pdfActionsDisabled = regeneratePdfMutation.isPending || pdfRegenBlocked;

  const editLockedMessage = isSigned
    ? "This contract is signed. Editing is locked for legal integrity. Use 'Mark as Unsigned' for correction."
    : null;

  const handleRegeneratePdf = async () => {
    try {
      setLocalError(null);
      await regeneratePdfMutation.mutateAsync();
    } catch (e) {
      console.error(e);
      setLocalError(e?.message || "Could not regenerate PDF");
    }
  };

  const handleSaveFields = async () => {
    try {
      setLocalError(null);
      const payload = {};
      for (const k of editableSet) {
        if (Object.prototype.hasOwnProperty.call(editFields, k)) {
          payload[k] = editFields[k];
        }
      }
      await updateFieldsMutation.mutateAsync(payload);
    } catch (e) {
      console.error(e);
      setLocalError(e?.message || "Could not save changes");
    }
  };

  const handleMarkSigned = async () => {
    try {
      setLocalError(null);
      await markSignedMutation.mutateAsync({
        agentName: signedByAgentName,
        clientName: signedByClientName,
      });
    } catch (e) {
      console.error(e);
      setLocalError(e?.message || "Could not mark as signed");
    }
  };

  const handleMarkUnsigned = async () => {
    try {
      setLocalError(null);
      await markUnsignedMutation.mutateAsync();
    } catch (e) {
      console.error(e);
      setLocalError(e?.message || "Could not mark as unsigned");
    }
  };

  const handleFieldChange = (key, value) => {
    setEditFields((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E]">
      <Header />

      <div className="pt-16">
        <div className="max-w-6xl mx-auto px-4 sm:px-8 py-8 sm:py-12">
          <ContractHeader
            propertyLink={propertyLink}
            templateType={contract.template_type}
            propertyTitle={contract.property_title}
            clientName={contract.client_name}
            pdfDownloadUrl={pdfDownloadUrl}
            onRegeneratePdf={handleRegeneratePdf}
            isRegenerating={regeneratePdfMutation.isPending}
            pdfActionsDisabled={pdfActionsDisabled}
            pdfRegenBlocked={pdfRegenBlocked}
          />

          {editLockedMessage ? (
            <div className="mb-6 rounded-lg bg-amber-50 dark:bg-amber-900/20 p-3 text-sm text-amber-800 dark:text-amber-200 font-jetbrains-mono">
              {editLockedMessage}
            </div>
          ) : null}

          {localError ? (
            <div className="mb-6 rounded-lg bg-red-50 dark:bg-red-900/30 p-3 text-sm text-red-600 dark:text-red-400 font-jetbrains-mono">
              {localError}
            </div>
          ) : null}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-1 space-y-6">
              <ContractFieldsEditor
                schema={schema}
                requiredSet={requiredSet}
                editableSet={editableSet}
                editFields={editFields}
                isSigned={isSigned}
                onFieldChange={handleFieldChange}
                onSave={handleSaveFields}
                isSaving={updateFieldsMutation.isPending}
                suggestionsMap={suggestionsMap}
              />

              <SignatureSection
                signatureMethod={signatureMethod}
                signedStatus={signedStatus}
                signedAt={signedAt}
                signedByClientName={signedByClientName}
                signedByAgentName={signedByAgentName}
                onClientNameChange={setSignedByClientName}
                onAgentNameChange={setSignedByAgentName}
                onMarkSigned={handleMarkSigned}
                onMarkUnsigned={handleMarkUnsigned}
                isMarkingSignedPending={markSignedMutation.isPending}
                isMarkingUnsignedPending={markUnsignedMutation.isPending}
              />

              <AuditSection auditEntries={auditEntries} />

              <FilledFieldsDebug filledFields={contract.filled_fields} />
            </div>

            <div className="lg:col-span-2">
              <PDFPreview
                pdfInlineUrl={pdfInlineUrl}
                pdfDownloadUrl={pdfDownloadUrl}
                pdfMessage={pdfMessage}
              />

              {pdfStatus === "failed" || pdfStatus === "disabled" ? (
                <div className="mt-4 rounded-lg bg-amber-50 dark:bg-amber-900/20 p-3 text-sm text-amber-800 dark:text-amber-200 font-jetbrains-mono">
                  PDF generation is temporarily unavailable for this contract.
                  The contract is saved; you can try again later.
                </div>
              ) : null}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
