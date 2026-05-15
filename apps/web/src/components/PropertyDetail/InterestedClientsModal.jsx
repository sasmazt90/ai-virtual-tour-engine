import { Check, Loader2, Search } from "lucide-react";
import { ModalShell } from "./ModalShell";

export function InterestedClientsModal({
  interestedModalOpen,
  setInterestedModalOpen,
  interestedSearch,
  setInterestedSearch,
  clientsLoading,
  filteredClients,
  selectedInterestedIds,
  setSelectedInterestedIds,
  saveInterestedMutation,
}) {
  if (!interestedModalOpen) return null;

  return (
    <ModalShell
      title="Add Interested Clients"
      onClose={() => setInterestedModalOpen(false)}
    >
      <div className="space-y-4">
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700">
          <Search size={16} className="text-gray-400" />
          <input
            value={interestedSearch}
            onChange={(e) => setInterestedSearch(e.target.value)}
            placeholder="Search clients..."
            className="w-full bg-transparent outline-none text-sm text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 font-jetbrains-mono"
          />
        </div>

        <div className="max-h-[360px] overflow-auto rounded-lg border border-gray-200 dark:border-gray-700">
          {clientsLoading ? (
            <div className="p-4 text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
              Loading clients...
            </div>
          ) : filteredClients.length === 0 ? (
            <div className="p-4 text-sm text-gray-600 dark:text-gray-400 font-jetbrains-mono">
              No matching clients.
            </div>
          ) : (
            filteredClients.map((c) => {
              const checked = selectedInterestedIds.includes(c.id);
              const subtitle = c.email || c.phone || "";
              return (
                <button
                  key={c.id}
                  type="button"
                  onClick={() => {
                    setSelectedInterestedIds((prev) => {
                      if (prev.includes(c.id)) {
                        return prev.filter((x) => x !== c.id);
                      }
                      return [...prev, c.id];
                    });
                  }}
                  className="w-full text-left px-4 py-3 border-b border-gray-200 dark:border-gray-700 last:border-b-0 hover:bg-gray-50 dark:hover:bg-gray-800"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="min-w-0">
                      <div className="font-medium text-gray-900 dark:text-gray-100 font-jetbrains-mono truncate">
                        {c.full_name}
                      </div>
                      {subtitle ? (
                        <div className="text-xs text-gray-500 dark:text-gray-400 font-jetbrains-mono truncate">
                          {subtitle}
                        </div>
                      ) : null}
                    </div>
                    {checked ? (
                      <Check size={18} className="text-[var(--brand)]" />
                    ) : null}
                  </div>
                </button>
              );
            })
          )}
        </div>

        <div className="flex items-center justify-end gap-3">
          <button
            onClick={() => setInterestedModalOpen(false)}
            className="px-4 py-2 rounded-lg border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-800 font-jetbrains-mono"
          >
            Cancel
          </button>
          <button
            onClick={() => saveInterestedMutation.mutate(selectedInterestedIds)}
            disabled={saveInterestedMutation.isPending}
            className="inline-flex items-center gap-2 px-5 py-2 rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 font-medium hover:bg-gray-800 dark:hover:bg-gray-200 disabled:opacity-50 font-jetbrains-mono"
          >
            {saveInterestedMutation.isPending ? (
              <Loader2 size={18} className="animate-spin" />
            ) : null}
            Save
          </button>
        </div>
      </div>
    </ModalShell>
  );
}
