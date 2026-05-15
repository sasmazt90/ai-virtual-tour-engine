import { Building2 } from "lucide-react";
import { formatNumberForInput } from "@/utils/formatters";
import {
  HOUSING_TYPE_OPTIONS,
  HOUSING_SHAPE_OPTIONS,
  HEATING_TYPE_OPTIONS,
  PARKING_OPTIONS,
  TITLE_DEED_OPTIONS,
  FURNISHED_OPTIONS,
  CONSTRUCTION_OPTIONS,
  USAGE_STATUS_OPTIONS,
  FACADE_OPTIONS,
} from "./constants";

export function PropertyDetailsSection({
  housingType,
  setHousingType,
  housingShape,
  setHousingShape,
  bedrooms,
  setBedrooms,
  livingRooms,
  setLivingRooms,
  bathrooms,
  setBathrooms,
  grossAreaInput,
  setGrossAreaInput,
  netAreaInput,
  setNetAreaInput,
  totalFloors,
  setTotalFloors,
  floorNumber,
  setFloorNumber,
  buildingAge,
  setBuildingAge,
  heatingType,
  setHeatingType,
  elevator,
  setElevator,
  parkingType,
  setParkingType,
  titleDeedStatus,
  setTitleDeedStatus,
  furnishedStatus,
  setFurnishedStatus,
  mortgageEligible,
  setMortgageEligible,
  constructionType,
  setConstructionType,
  usageStatus,
  setUsageStatus,
  facade,
  setFacade,
  description,
  setDescription,
}) {
  return (
    <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6 sm:p-8">
      <div className="flex items-center gap-3 mb-6">
        <Building2 className="text-[var(--brand)]" />
        <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          Property Details
        </h2>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Housing type
          </label>
          <select
            value={housingType}
            onChange={(e) => setHousingType(e.target.value)}
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
          >
            <option value="">Select</option>
            {HOUSING_TYPE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Layout
          </label>
          <select
            value={housingShape}
            onChange={(e) => setHousingShape(e.target.value)}
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
          >
            <option value="">Select</option>
            {HOUSING_SHAPE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Bedrooms
          </label>
          <input
            value={bedrooms}
            onChange={(e) => setBedrooms(e.target.value)}
            inputMode="numeric"
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
            placeholder="e.g. 2"
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Living rooms
          </label>
          <input
            value={livingRooms}
            onChange={(e) => setLivingRooms(e.target.value)}
            inputMode="numeric"
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
            placeholder="e.g. 1"
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Bathrooms
          </label>
          <input
            value={bathrooms}
            onChange={(e) => setBathrooms(e.target.value)}
            inputMode="numeric"
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
            placeholder="e.g. 1"
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Gross area (m²)
          </label>
          <input
            value={grossAreaInput}
            onChange={(e) => setGrossAreaInput(e.target.value)}
            onBlur={() =>
              setGrossAreaInput(formatNumberForInput(grossAreaInput))
            }
            inputMode="decimal"
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
            placeholder="e.g. 120"
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Net area (m²)
          </label>
          <input
            value={netAreaInput}
            onChange={(e) => setNetAreaInput(e.target.value)}
            onBlur={() => setNetAreaInput(formatNumberForInput(netAreaInput))}
            inputMode="decimal"
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
            placeholder="e.g. 95"
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Total floors
          </label>
          <input
            value={totalFloors}
            onChange={(e) => setTotalFloors(e.target.value)}
            inputMode="numeric"
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
            placeholder="e.g. 10"
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Floor number
          </label>
          <input
            value={floorNumber}
            onChange={(e) => setFloorNumber(e.target.value)}
            inputMode="numeric"
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
            placeholder="e.g. 3"
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Building age
          </label>
          <input
            value={buildingAge}
            onChange={(e) => setBuildingAge(e.target.value)}
            inputMode="numeric"
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
            placeholder="e.g. 5"
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Heating type
          </label>
          <select
            value={heatingType}
            onChange={(e) => setHeatingType(e.target.value)}
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
          >
            <option value="">Select</option>
            {HEATING_TYPE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Elevator
          </label>
          <select
            value={elevator}
            onChange={(e) => setElevator(e.target.value)}
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
          >
            <option value="">Select</option>
            <option value="yes">Yes</option>
            <option value="no">No</option>
          </select>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Parking
          </label>
          <select
            value={parkingType}
            onChange={(e) => setParkingType(e.target.value)}
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
          >
            <option value="">Select</option>
            {PARKING_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Title deed status
          </label>
          <select
            value={titleDeedStatus}
            onChange={(e) => setTitleDeedStatus(e.target.value)}
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
          >
            <option value="">Select</option>
            {TITLE_DEED_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Furnishing
          </label>
          <select
            value={furnishedStatus}
            onChange={(e) => setFurnishedStatus(e.target.value)}
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
          >
            <option value="">Select</option>
            {FURNISHED_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Mortgage eligible
          </label>
          <select
            value={mortgageEligible}
            onChange={(e) => setMortgageEligible(e.target.value)}
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
          >
            <option value="">Select</option>
            <option value="yes">Yes</option>
            <option value="no">No</option>
          </select>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Construction type
          </label>
          <select
            value={constructionType}
            onChange={(e) => setConstructionType(e.target.value)}
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
          >
            <option value="">Select</option>
            {CONSTRUCTION_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Usage status
          </label>
          <select
            value={usageStatus}
            onChange={(e) => setUsageStatus(e.target.value)}
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
          >
            <option value="">Select</option>
            {USAGE_STATUS_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Facade
          </label>
          <select
            value={facade}
            onChange={(e) => setFacade(e.target.value)}
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
          >
            <option value="">Select</option>
            {FACADE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div className="sm:col-span-2 space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Description
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={5}
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
            placeholder="Short description for your listing"
          />
        </div>
      </div>
    </div>
  );
}
