export function StatusBanner({ variant = "error", title = null, children }) {
  const styles =
    variant === "error"
      ? "bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-300"
      : variant === "warning"
        ? "bg-amber-50 dark:bg-amber-900/20 text-amber-800 dark:text-amber-200"
        : variant === "success"
          ? "bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-200"
          : "bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-200";

  return (
    <div
      className={`rounded-lg px-4 py-3 text-sm font-jetbrains-mono ${styles}`}
    >
      {title ? <div className="font-semibold">{title}</div> : null}
      {children ? <div className={title ? "mt-1" : ""}>{children}</div> : null}
    </div>
  );
}
