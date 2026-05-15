import { useCallback, useState } from "react";
// NOTE: This file is the canonical page entry for the /properties/[id] route.
// Splitting UI into smaller pieces (under /components/PropertyDetail/*) is OK,
// but keep routing and data fetching centralized here to avoid accidental route splits
// or duplicated fetch logic.
import { useQueryClient } from "@tanstack/react-query";
import useUser from "@/utils/useUser";
import { Header } from "@/components/Header";
import { ArrowLeft } from "lucide-react";
import { usePropertyData } from "@/hooks/usePropertyData";
import { useCredits } from "@/hooks/useCredits";
import { useInterestedClients } from "@/hooks/useInterestedClients";
import { useStagingHelpers } from "@/hooks/useStagingHelpers";
import { PropertyHeader } from "@/components/PropertyDetail/PropertyHeader";
import { PropertyOverview } from "@/components/PropertyDetail/PropertyOverview";
import { PropertyPhotos } from "@/components/PropertyDetail/PropertyPhotos";
import { PropertyAssets } from "@/components/PropertyDetail/PropertyAssets";
import { PropertyOwner } from "@/components/PropertyDetail/PropertyOwner";
import { InterestedClientsCard } from "@/components/PropertyDetail/InterestedClientsCard";
import { InterestedClientsModal } from "@/components/PropertyDetail/InterestedClientsModal";
import CreateStagingModal from "@/components/PropertyDetail/CreateStagingModal";
import { SharePropertyModal } from "@/components/PropertyDetail/SharePropertyModal";

// (AI Studio removed; pricing calculations now live inside the staging popup)

export default function PropertyDetailPage(props) {
  const propertyId = props?.params?.id;
  const queryClient = useQueryClient();
  const { data: user, loading: userLoading } = useUser();

  const { property, propertyLoading, propertyError, refetchProperty } =
    usePropertyData(user?.id, propertyId);

  const { creditsBalance } = useCredits(user?.id);

  const interestedClients = property?.interested_clients || [];

  const {
    interestedModalOpen,
    setInterestedModalOpen,
    interestedSearch,
    setInterestedSearch,
    selectedInterestedIds,
    setSelectedInterestedIds,
    clientsLoading,
    filteredClients,
    openInterestedModal,
    saveInterestedMutation,
    removeInterestedClient,
  } = useInterestedClients(
    user?.id,
    propertyId,
    interestedClients,
    refetchProperty,
  );

  const { formatStagingLabel } = useStagingHelpers(property);

  const onRefreshAfterJob = useCallback(async () => {
    await refetchProperty();
    queryClient.invalidateQueries({ queryKey: ["credits", user?.id] });
  }, [queryClient, refetchProperty, user?.id]);

  const onRefreshAssets = useCallback(async () => {
    await refetchProperty();
  }, [refetchProperty]);

  const [stagingOpen, setStagingOpen] = useState(false);
  const onOpenStaging = useCallback(() => setStagingOpen(true), []);
  const onCloseStaging = useCallback(() => setStagingOpen(false), []);

  const [shareOpen, setShareOpen] = useState(false);
  const [shareInitialClientId, setShareInitialClientId] = useState(null);
  const onCloseShare = useCallback(() => setShareOpen(false), []);

  const onOpenShareForClient = useCallback((clientId) => {
    if (typeof clientId === "string" && clientId.trim()) {
      setShareInitialClientId(clientId.trim());
    } else {
      setShareInitialClientId(null);
    }
    setShareOpen(true);
  }, []);

  const onOpenShareNoClient = useCallback(() => {
    setShareInitialClientId(null);
    setShareOpen(true);
  }, []);

  if (userLoading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E] ui-surface">
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

  if (propertyLoading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E] ui-surface">
        <Header />
        <div className="pt-16 flex items-center justify-center min-h-screen">
          <p className="text-gray-600 dark:text-gray-400 font-jetbrains-mono">
            Loading property...
          </p>
        </div>
      </div>
    );
  }

  if (propertyError || !property) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E] ui-surface">
        <Header />
        <div className="pt-16 max-w-4xl mx-auto px-4 sm:px-8 py-12">
          <a
            href="/properties"
            className="inline-flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 font-jetbrains-mono"
          >
            <ArrowLeft size={16} />
            Back to Properties
          </a>
          <div className="mt-6 rounded-xl bg-white dark:bg-[#262626] p-6 shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700">
            <p className="text-red-600 dark:text-red-400 font-jetbrains-mono">
              Could not load this property.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const photos = property.photos || [];
  const ownerName = property.owner_name || "—";
  const ownerEmail = property.owner_email || null;
  const ownerPhone = property.owner_phone || null;

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E] ui-surface">
      <Header />

      <div className="pt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-8 py-8 sm:py-12">
          <PropertyHeader
            property={property}
            propertyId={propertyId}
            onOpenShare={onOpenShareNoClient}
          />

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 space-y-8">
              <PropertyOverview property={property} />
              {/* pass propertyId/userId so photos can be added/removed */}
              <PropertyPhotos
                photos={photos}
                propertyId={propertyId}
                userId={user?.id}
              />
              <PropertyAssets
                property={property}
                propertyId={propertyId}
                formatStagingLabel={formatStagingLabel}
                onOpenShareForClient={onOpenShareForClient}
                onAddStaging={onOpenStaging}
                onRefresh={onRefreshAssets}
              />
            </div>

            <div className="space-y-8">
              <PropertyOwner
                ownerName={ownerName}
                ownerEmail={ownerEmail}
                ownerPhone={ownerPhone}
              />
              <InterestedClientsCard
                interestedClients={interestedClients}
                openInterestedModal={openInterestedModal}
                removeInterestedClient={removeInterestedClient}
              />
            </div>
          </div>
        </div>
      </div>

      <InterestedClientsModal
        interestedModalOpen={interestedModalOpen}
        setInterestedModalOpen={setInterestedModalOpen}
        interestedSearch={interestedSearch}
        setInterestedSearch={setInterestedSearch}
        clientsLoading={clientsLoading}
        filteredClients={filteredClients}
        selectedInterestedIds={selectedInterestedIds}
        setSelectedInterestedIds={setSelectedInterestedIds}
        saveInterestedMutation={saveInterestedMutation}
      />

      <CreateStagingModal
        open={stagingOpen}
        onClose={onCloseStaging}
        userId={user?.id}
        propertyId={propertyId}
        property={property}
        creditsBalance={creditsBalance}
        onRefreshAfterJob={onRefreshAfterJob}
      />

      <SharePropertyModal
        open={shareOpen}
        onClose={onCloseShare}
        property={property}
        initialClientId={shareInitialClientId}
      />
    </div>
  );
}
