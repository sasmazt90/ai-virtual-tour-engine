export function PropertyOwner({ ownerName, ownerEmail, ownerPhone }) {
  return (
    <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6">
      <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 font-jetbrains-mono">
        Owner
      </h2>
      <div className="space-y-2 text-sm">
        <div className="font-medium text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          {ownerName}
        </div>
        {ownerEmail ? (
          <div className="text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            {ownerEmail}
          </div>
        ) : null}
        {ownerPhone ? (
          <div className="text-gray-600 dark:text-gray-300 font-jetbrains-mono">
            {ownerPhone}
          </div>
        ) : null}
      </div>
    </div>
  );
}
