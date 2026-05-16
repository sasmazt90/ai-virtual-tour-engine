export function CreditEstimate({
  stagingCostPerPhoto,
  stagingCostPerPhotoWithCustomFurniture,
  estimatedTotalCost,
  missingCreditsText,
  ackCredits,
  onAckChange,
  disabled,
}) {
  return (
    <div className="mt-4 rounded-lg border border-gray-200 dark:border-gray-700 p-4 bg-gray-50 dark:bg-gray-900">
      <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
        Estimated credit cost
      </div>
      <div className="mt-2 text-sm text-gray-700 dark:text-gray-200 font-jetbrains-mono">
        Staging (Vacant): {Number(stagingCostPerPhoto).toLocaleString()} •
        Staging (Luxury): {Number(stagingCostPerPhoto).toLocaleString()} •
        Staging (Modern + Custom):{" "}
        {Number(stagingCostPerPhotoWithCustomFurniture).toLocaleString()}
      </div>
      <div className="mt-2 text-sm text-gray-900 dark:text-gray-100 font-jetbrains-mono">
        Total: {Number(estimatedTotalCost || 0).toLocaleString()} credits
      </div>

      {missingCreditsText ? (
        <div className="mt-2 text-xs text-red-600 dark:text-red-400 font-jetbrains-mono">
          {missingCreditsText}
        </div>
      ) : null}

      <label className="mt-3 flex items-start gap-2 text-xs text-gray-600 dark:text-gray-300 font-jetbrains-mono">
        <input
          type="checkbox"
          checked={ackCredits}
          onChange={(e) => onAckChange(e.target.checked)}
          disabled={disabled}
          className="mt-0.5"
        />
        <span>
          I understand running these tests will spend credits and call OpenAI.
        </span>
      </label>
    </div>
  );
}
