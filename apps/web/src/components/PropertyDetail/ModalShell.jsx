import { X } from "lucide-react";

export function ModalShell({ title, onClose, children, headerActions }) {
  return (
    <div className="fixed inset-0 z-50 bg-black/50 overflow-y-auto">
      <div className="min-h-full w-full flex items-start sm:items-center justify-center p-2 sm:p-4">
        <div className="w-full max-w-3xl max-h-[calc(100vh-2rem)] rounded-2xl bg-white dark:bg-[#262626] shadow-xl dark:shadow-none dark:ring-1 dark:ring-gray-700 overflow-hidden flex flex-col">
          <div className="flex items-center justify-between px-4 sm:px-6 py-4 border-b border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 font-jetbrains-mono">
              {title}
            </h3>

            <div className="flex items-center gap-2">
              {headerActions ? <div>{headerActions}</div> : null}
              <button
                onClick={onClose}
                className="text-gray-500 hover:text-gray-700 dark:text-gray-300 dark:hover:text-gray-100"
                aria-label="Close"
              >
                <X size={20} />
              </button>
            </div>
          </div>
          <div className="p-4 sm:p-6 overflow-y-auto">{children}</div>
        </div>
      </div>
    </div>
  );
}
