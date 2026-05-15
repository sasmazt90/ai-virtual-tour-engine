import { Building2 } from "lucide-react";
import { CheckboxGrid } from "./CheckboxGrid";
import {
  FEATURES_INTERIOR_GROUPS,
  FEATURES_EXTERIOR_GROUPS,
} from "./constants";

export function FeaturesSection({
  featuresInterior,
  setFeaturesInterior,
  featuresExterior,
  setFeaturesExterior,
}) {
  return (
    <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6 sm:p-8">
      <div className="flex items-center gap-3 mb-6">
        <Building2 className="text-[var(--brand)]" />
        <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          Features
        </h2>
      </div>

      <div className="space-y-4">
        <details className="rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <summary className="cursor-pointer text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Interior features
          </summary>
          <div className="mt-3">
            <CheckboxGrid
              options={FEATURES_INTERIOR_GROUPS}
              selected={featuresInterior}
              setSelected={setFeaturesInterior}
            />
          </div>
        </details>

        <details className="rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <summary className="cursor-pointer text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
            Exterior features
          </summary>
          <div className="mt-3">
            <CheckboxGrid
              options={FEATURES_EXTERIOR_GROUPS}
              selected={featuresExterior}
              setSelected={setFeaturesExterior}
            />
          </div>
        </details>
      </div>
    </div>
  );
}
