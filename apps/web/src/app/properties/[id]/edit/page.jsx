import { useQuery } from "@tanstack/react-query";
import useUser from "@/utils/useUser";
import { Header } from "@/components/Header";
import { Loader2, ArrowLeft } from "lucide-react";
import { usePropertyData } from "@/hooks/usePropertyData";
import { useEditPropertyForm } from "@/hooks/useEditPropertyForm";
import { PropertyInfoSection } from "@/components/PropertyForm/PropertyInfoSection";
import { AddressSection } from "@/components/PropertyForm/AddressSection";
import { PropertyDetailsSection } from "@/components/PropertyForm/PropertyDetailsSection";
import { FeaturesSection } from "@/components/PropertyForm/FeaturesSection";
import { OwnerInfoSection } from "@/components/PropertyForm/OwnerInfoSection";

export default function EditPropertyPage(props) {
  const propertyId = props?.params?.id;
  const { data: user, loading: userLoading } = useUser();

  const { property, propertyLoading, propertyError } = usePropertyData(
    user?.id,
    propertyId,
  );

  const formState = useEditPropertyForm({ user, propertyId, property });

  const { data: clients = [], isLoading: clientsLoading } = useQuery({
    queryKey: ["clients", user?.id],
    queryFn: async () => {
      const res = await fetch("/api/clients");
      if (!res.ok) {
        throw new Error(
          `When fetching /api/clients, the response was [${res.status}] ${res.statusText}`,
        );
      }
      return res.json();
    },
    enabled: !!user?.id,
  });

  const submitLabel = formState.submitting ? "Saving..." : "Save Changes";

  if (userLoading) {
    return (
      <div className="min-h-screen ui-surface">
        <Header />
        <div className="pt-16 flex items-center justify-center min-h-screen">
          <p className="text-gray-700 dark:text-gray-300 font-jetbrains-mono">
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
      <div className="min-h-screen ui-surface">
        <Header />
        <div className="pt-16 flex items-center justify-center min-h-screen">
          <p className="text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Loading property...
          </p>
        </div>
      </div>
    );
  }

  if (propertyError || !property) {
    return (
      <div className="min-h-screen ui-surface">
        <Header />
        <div className="pt-16 max-w-5xl mx-auto px-4 sm:px-8 py-8 sm:py-12">
          <a
            href={propertyId ? `/properties/${propertyId}` : "/properties"}
            className="inline-flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 font-jetbrains-mono"
          >
            <ArrowLeft size={16} />
            Back
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

  return (
    <div className="min-h-screen ui-surface">
      <Header />

      <div className="pt-16">
        <div className="max-w-5xl mx-auto px-4 sm:px-8 py-8 sm:py-12">
          <div className="mb-8">
            <a
              href={`/properties/${propertyId}`}
              className="inline-flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 font-jetbrains-mono"
            >
              <ArrowLeft size={16} />
              Back to Property
            </a>

            <h1 className="mt-4 text-3xl sm:text-4xl font-bold text-gray-900 dark:text-gray-100 mb-2 font-jetbrains-mono">
              Edit Property
            </h1>
            <p className="text-gray-600 dark:text-gray-300 font-jetbrains-mono">
              Update the listing details.
            </p>
          </div>

          <form onSubmit={formState.onSubmit} className="space-y-8">
            <PropertyInfoSection
              title={formState.title}
              setTitle={formState.setTitle}
              propertyStatus={formState.propertyStatus}
              setPropertyStatus={formState.setPropertyStatus}
              currency={formState.currency}
              setCurrency={formState.setCurrency}
              priceInput={formState.priceInput}
              setPriceInput={formState.setPriceInput}
              depositInput={formState.depositInput}
              setDepositInput={formState.setDepositInput}
              duesInput={formState.duesInput}
              setDuesInput={formState.setDuesInput}
            />

            <AddressSection
              addressLine={formState.addressLine}
              setAddressLine={formState.setAddressLine}
              city={formState.city}
              setCity={formState.setCity}
              postalCode={formState.postalCode}
              setPostalCode={formState.setPostalCode}
              country={formState.country}
              setCountry={formState.setCountry}
            />

            <PropertyDetailsSection
              housingType={formState.housingType}
              setHousingType={formState.setHousingType}
              housingShape={formState.housingShape}
              setHousingShape={formState.setHousingShape}
              bedrooms={formState.bedrooms}
              setBedrooms={formState.setBedrooms}
              livingRooms={formState.livingRooms}
              setLivingRooms={formState.setLivingRooms}
              bathrooms={formState.bathrooms}
              setBathrooms={formState.setBathrooms}
              grossAreaInput={formState.grossAreaInput}
              setGrossAreaInput={formState.setGrossAreaInput}
              netAreaInput={formState.netAreaInput}
              setNetAreaInput={formState.setNetAreaInput}
              totalFloors={formState.totalFloors}
              setTotalFloors={formState.setTotalFloors}
              floorNumber={formState.floorNumber}
              setFloorNumber={formState.setFloorNumber}
              buildingAge={formState.buildingAge}
              setBuildingAge={formState.setBuildingAge}
              heatingType={formState.heatingType}
              setHeatingType={formState.setHeatingType}
              elevator={formState.elevator}
              setElevator={formState.setElevator}
              parkingType={formState.parkingType}
              setParkingType={formState.setParkingType}
              titleDeedStatus={formState.titleDeedStatus}
              setTitleDeedStatus={formState.setTitleDeedStatus}
              furnishedStatus={formState.furnishedStatus}
              setFurnishedStatus={formState.setFurnishedStatus}
              mortgageEligible={formState.mortgageEligible}
              setMortgageEligible={formState.setMortgageEligible}
              constructionType={formState.constructionType}
              setConstructionType={formState.setConstructionType}
              usageStatus={formState.usageStatus}
              setUsageStatus={formState.setUsageStatus}
              facade={formState.facade}
              setFacade={formState.setFacade}
              description={formState.description}
              setDescription={formState.setDescription}
            />

            <FeaturesSection
              featuresInterior={formState.featuresInterior}
              setFeaturesInterior={formState.setFeaturesInterior}
              featuresExterior={formState.featuresExterior}
              setFeaturesExterior={formState.setFeaturesExterior}
            />

            <OwnerInfoSection
              ownerMode={formState.ownerMode}
              setOwnerMode={formState.setOwnerMode}
              ownerSearch={formState.ownerSearch}
              setOwnerSearch={formState.setOwnerSearch}
              ownerDropdownOpen={formState.ownerDropdownOpen}
              setOwnerDropdownOpen={formState.setOwnerDropdownOpen}
              selectedOwnerClientId={formState.selectedOwnerClientId}
              setSelectedOwnerClientId={formState.setSelectedOwnerClientId}
              newOwnerName={formState.newOwnerName}
              setNewOwnerName={formState.setNewOwnerName}
              newOwnerEmail={formState.newOwnerEmail}
              setNewOwnerEmail={formState.setNewOwnerEmail}
              newOwnerPhone={formState.newOwnerPhone}
              setNewOwnerPhone={formState.setNewOwnerPhone}
              newOwnerCountryCode={formState.newOwnerCountryCode}
              setNewOwnerCountryCode={formState.setNewOwnerCountryCode}
              newOwnerCity={formState.newOwnerCity}
              setNewOwnerCity={formState.setNewOwnerCity}
              clients={clients}
              clientsLoading={clientsLoading}
              resetOwnerSelection={formState.resetOwnerSelection}
            />

            {formState.error ? (
              <div className="rounded-lg bg-red-50 dark:bg-red-900/30 p-3 text-sm text-red-600 dark:text-red-400 font-jetbrains-mono">
                {formState.error}
              </div>
            ) : null}

            <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center justify-between">
              <a
                href={`/properties/${propertyId}`}
                className="text-sm text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 font-jetbrains-mono"
              >
                Cancel
              </a>

              <button
                type="submit"
                disabled={formState.submitting}
                className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-[var(--brand90)] hover:bg-[var(--brand)] text-white rounded-lg font-medium transition-colors disabled:opacity-50 font-jetbrains-mono"
              >
                {formState.submitting ? (
                  <Loader2 size={18} className="animate-spin" />
                ) : null}
                {submitLabel}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
