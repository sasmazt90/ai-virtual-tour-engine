import { useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import useUser from "@/utils/useUser";
import { Header } from "@/components/Header";
import { StatusBanner } from "@/components/StatusBanner";
import { useCredits } from "@/hooks/useCredits";
import { calculateStagingCreditCost } from "@/app/api/utils/pricing";
import useUpload from "@/utils/useUpload";
import { PropertySelector } from "@/components/E2ETests/PropertySelector";
import { PropertyStats } from "@/components/E2ETests/PropertyStats";
import { CreditEstimate } from "@/components/E2ETests/CreditEstimate";
import { CustomFurnitureUpload } from "@/components/E2ETests/CustomFurnitureUpload";
import { TestResults } from "@/components/E2ETests/TestResults";
import { VacantQuickTest } from "@/components/E2ETests/VacantQuickTest";
import { usePropertyData } from "@/hooks/usePropertyData";
import { useE2ETests } from "@/hooks/useE2ETests";
import { useVacantQuickTest } from "@/hooks/useVacantQuickTest";

export default function AIE2ETestsPage() {
  const queryClient = useQueryClient();
  const { data: user, loading: userLoading } = useUser();
  const { creditsBalance } = useCredits(user?.id);

  const [selectedPropertyId, setSelectedPropertyId] = useState("");
  const [ackCredits, setAckCredits] = useState(false);
  const [furnitureFile, setFurnitureFile] = useState(null);

  const [upload, { loading: uploadingFurniture }] = useUpload();

  const {
    properties,
    propertiesLoading,
    selectedProperty,
    selectedPropertyPhotoIds,
    firstPhotoId,
    customAssetIds,
    hasCustomAssets,
  } = usePropertyData(user?.id, selectedPropertyId);

  const stagingCostPerPhoto = useMemo(() => {
    return calculateStagingCreditCost({
      hasPreferredItems: false,
      hasCustomAssets: false,
      photoCount: 1,
    });
  }, []);

  const stagingCostPerPhotoWithCustomFurniture = useMemo(() => {
    return calculateStagingCreditCost({
      hasPreferredItems: false,
      hasCustomAssets: true,
      photoCount: 1,
    });
  }, []);

  const estimatedTotalCost = useMemo(() => {
    const baseStaging = Number(stagingCostPerPhoto || 0) * 2;
    const modernWithFurniture = Number(stagingCostPerPhotoWithCustomFurniture);
    return baseStaging + modernWithFurniture;
  }, [stagingCostPerPhoto, stagingCostPerPhotoWithCustomFurniture]);

  const { e2eRunning, e2eError, results, runE2E } = useE2ETests({
    selectedPropertyId,
    firstPhotoId,
    customAssetIds,
    hasCustomAssets,
    furnitureFile,
    upload,
    queryClient,
  });

  const { vacantQuick, runVacantQuickTest } = useVacantQuickTest({
    selectedPropertyId,
    firstPhotoId,
    queryClient,
  });

  const canRun =
    !e2eRunning &&
    !!selectedPropertyId &&
    !!firstPhotoId &&
    ackCredits &&
    Number(creditsBalance || 0) >= Number(estimatedTotalCost || 0);

  const canRunVacantQuick =
    !e2eRunning &&
    vacantQuick.status !== "running" &&
    vacantQuick.status !== "queued" &&
    !!selectedPropertyId &&
    !!firstPhotoId;

  if (userLoading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E]">
        <Header />
        <div className="pt-16 max-w-4xl mx-auto px-4 sm:px-8 py-12">
          <div className="text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
            Loading…
          </div>
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

  const propertySelectDisabled = e2eRunning || propertiesLoading;

  const hasEnoughCredits =
    Number(creditsBalance || 0) >= Number(estimatedTotalCost || 0);

  const missingCreditsText = hasEnoughCredits
    ? null
    : `Not enough credits for this full test run. Needed: ${Number(estimatedTotalCost || 0).toLocaleString()}, you have: ${Number(creditsBalance || 0).toLocaleString()}.`;

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-[#1E1E1E]">
      <Header />

      <div className="pt-16">
        <div className="max-w-4xl mx-auto px-4 sm:px-8 py-8 sm:py-12">
          <a
            href="/profile/tools"
            className="inline-flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 font-jetbrains-mono"
          >
            <ArrowLeft size={16} />
            Back to Tools
          </a>

          <div className="mt-6 mb-6">
            <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
              AI E2E Tests
            </h1>
            <p className="mt-2 text-sm text-gray-600 dark:text-gray-300 font-jetbrains-mono">
              This page runs the same staging flows you run in the UI, but in a
              guided, repeatable sequence.
            </p>
            <p className="mt-2 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
              Note: I can't log in as your account from here, so this is the
              safest way to run an end-to-end test using your own session.
            </p>
          </div>

          <PropertySelector
            selectedPropertyId={selectedPropertyId}
            onPropertyChange={setSelectedPropertyId}
            properties={properties}
            disabled={propertySelectDisabled}
          />

          {selectedProperty ? (
            <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-5 mt-4">
              <PropertyStats
                photoCount={selectedPropertyPhotoIds.length}
                customAssetCount={customAssetIds.length}
                creditsBalance={creditsBalance}
              />

              <CreditEstimate
                stagingCostPerPhoto={stagingCostPerPhoto}
                stagingCostPerPhotoWithCustomFurniture={
                  stagingCostPerPhotoWithCustomFurniture
                }
                estimatedTotalCost={estimatedTotalCost}
                missingCreditsText={missingCreditsText}
                ackCredits={ackCredits}
                onAckChange={setAckCredits}
                disabled={e2eRunning}
              />

              <CustomFurnitureUpload
                furnitureFile={furnitureFile}
                onFileChange={setFurnitureFile}
                disabled={e2eRunning}
                uploading={uploadingFurniture}
              />

              <div className="mt-4 flex items-center justify-end gap-2">
                <a
                  href={
                    selectedPropertyId
                      ? `/properties/${selectedPropertyId}`
                      : "/properties"
                  }
                  className="inline-flex items-center px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-200 text-sm font-medium hover:bg-gray-50 dark:hover:bg-gray-800 font-jetbrains-mono"
                >
                  Open property
                </a>
                <button
                  type="button"
                  onClick={runE2E}
                  disabled={!canRun}
                  className="inline-flex items-center px-4 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:bg-gray-800 dark:hover:bg-gray-200 disabled:opacity-50 font-jetbrains-mono"
                >
                  {e2eRunning ? "Running…" : "Run 4 tests"}
                </button>
              </div>

              {e2eError ? (
                <div className="mt-4">
                  <StatusBanner variant="error">{e2eError}</StatusBanner>
                </div>
              ) : null}

              <VacantQuickTest
                vacantQuick={vacantQuick}
                onRun={runVacantQuickTest}
                canRun={canRunVacantQuick}
              />
            </div>
          ) : null}

          <TestResults results={results} />

          <div className="mt-6 text-xs text-gray-500 dark:text-gray-500 font-jetbrains-mono">
            Tip: after a pass, open the property and check Assets → Stagings /
            Virtual Tour. Each staging test uses the first property photo to
            keep cost low.
          </div>
        </div>
      </div>
    </div>
  );
}
