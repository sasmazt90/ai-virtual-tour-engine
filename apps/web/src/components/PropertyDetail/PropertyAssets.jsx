import { useMemo } from "react";
import { StagingsSection } from "./StagingsSection";
import { VirtualToursSection } from "./VirtualToursSection";
import { ContractsSection } from "./ContractsSection";

export function PropertyAssets({
  property,
  propertyId,
  formatStagingLabel,
  onOpenShareForClient,
  onAddStaging,
  onRefresh,
}) {
  // Client context for inline share management: explicit selection (no default)
  const ownerClient = useMemo(() => {
    if (property?.owner_client_id && property?.owner_name) {
      return { id: property.owner_client_id, label: property.owner_name };
    }
    return null;
  }, [property]);

  const interestedClients = useMemo(() => {
    const interested = Array.isArray(property?.interested_clients)
      ? property.interested_clients
      : [];
    const list = [];
    for (const c of interested) {
      if (!c?.id) continue;
      if (ownerClient?.id && c.id === ownerClient.id) continue;
      list.push({ id: c.id, label: c.full_name || "Client" });
    }
    return list;
  }, [ownerClient?.id, property]);

  return (
    <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6">
      <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-6 font-jetbrains-mono">
        Assets
      </h2>

      <div className="space-y-8">
        <StagingsSection
          stagings={property.stagings}
          formatStagingLabel={formatStagingLabel}
          onAddNew={onAddStaging}
          onRefresh={onRefresh}
        />

        <VirtualToursSection property={property} propertyId={propertyId} />

        <ContractsSection
          property={property}
          propertyId={propertyId}
          ownerClient={ownerClient}
          interestedClients={interestedClients}
          onOpenShareForClient={onOpenShareForClient}
        />
      </div>
    </div>
  );
}
