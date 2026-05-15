import { Building2 } from "lucide-react";
import { formatIntegerForInput } from "@/utils/formatters";
import { STATUS_OPTIONS, CURRENCY_OPTIONS } from "./constants";

export function PropertyInfoSection({
  title,
  setTitle,
  propertyStatus,
  setPropertyStatus,
  currency,
  setCurrency,
  priceInput,
  setPriceInput,
  depositInput,
  setDepositInput,
  duesInput,
  setDuesInput,
}) {
  return (
    <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6 sm:p-8">
      <div className="flex items-center gap-3 mb-6">
        <Building2 className="text-[var(--brand)]" />
        <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          Property Info
        </h2>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
        <div className="sm:col-span-2 space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Title
          </label>
          <input
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
            placeholder="e.g. Sunny 2-bedroom downtown"
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Status
          </label>
          <select
            value={propertyStatus}
            onChange={(e) => setPropertyStatus(e.target.value)}
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
          >
            {STATUS_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Currency
          </label>
          <select
            value={currency}
            onChange={(e) => setCurrency(e.target.value)}
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
          >
            {CURRENCY_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Price
          </label>
          <input
            value={priceInput}
            onChange={(e) => setPriceInput(e.target.value)}
            onBlur={() => setPriceInput(formatIntegerForInput(priceInput))}
            inputMode="numeric"
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
            placeholder="e.g. 450,000"
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Deposit
          </label>
          <input
            value={depositInput}
            onChange={(e) => setDepositInput(e.target.value)}
            onBlur={() => setDepositInput(formatIntegerForInput(depositInput))}
            inputMode="numeric"
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
            placeholder="e.g. 50,000"
          />
        </div>

        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 font-jetbrains-mono">
            Dues
          </label>
          <input
            value={duesInput}
            onChange={(e) => setDuesInput(e.target.value)}
            onBlur={() => setDuesInput(formatIntegerForInput(duesInput))}
            inputMode="numeric"
            className="w-full px-4 py-3 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--brand)] font-jetbrains-mono"
            placeholder="e.g. 2,500"
          />
        </div>
      </div>
    </div>
  );
}
