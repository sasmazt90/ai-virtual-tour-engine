import { Plus, X } from "lucide-react";

export function InterestedClientsCard({
  interestedClients,
  openInterestedModal,
  removeInterestedClient,
}) {
  return (
    <div className="bg-white dark:bg-[#262626] rounded-xl shadow-lg dark:shadow-none dark:ring-1 dark:ring-gray-700 p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
          Interested Clients
        </h2>
        <button
          onClick={openInterestedModal}
          className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 text-sm font-medium hover:bg-gray-800 dark:hover:bg-gray-200 transition-colors font-jetbrains-mono"
        >
          <Plus size={16} />
          Add
        </button>
      </div>

      {interestedClients.length === 0 ? (
        <div className="text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
          None yet.
        </div>
      ) : (
        <div className="space-y-2">
          {interestedClients.map((c) => (
            <div
              key={c.id}
              className="flex items-center justify-between gap-3 rounded-lg border border-gray-200 dark:border-gray-700 px-3 py-2"
            >
              <div className="min-w-0">
                <div className="text-sm font-medium text-gray-900 dark:text-gray-100 font-jetbrains-mono truncate">
                  {c.full_name}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono truncate">
                  {c.email || c.phone || ""}
                </div>
              </div>
              <button
                onClick={() => removeInterestedClient(c.id)}
                className="text-gray-400 hover:text-red-600 dark:hover:text-red-400"
                aria-label="Remove"
              >
                <X size={16} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
