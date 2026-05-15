import { useCallback, useMemo } from "react";
import { titleCase } from "@/utils/formatters";

export function useStagingHelpers(property) {
  const latestVersionByType = useMemo(() => {
    const map = {};
    for (const s of property?.stagings || []) {
      const t = s.staging_type;
      const v = Number(s.version || 1);
      if (!map[t] || v > map[t]) map[t] = v;
    }
    return map;
  }, [property?.stagings]);

  const formatStagingLabel = useCallback(
    (s) => {
      const t = titleCase(s?.staging_type || "staging");
      const v = Number(s?.version || 1);
      const isLatest = latestVersionByType[s?.staging_type] === v;
      return `${t} (v${v}${isLatest ? " – latest" : ""})`;
    },
    [latestVersionByType],
  );

  return { formatStagingLabel };
}
