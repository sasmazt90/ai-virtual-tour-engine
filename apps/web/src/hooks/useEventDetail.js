import { useState, useCallback } from "react";

export function useEventDetail() {
  const [detailOpen, setDetailOpen] = useState(false);
  const [detailEvent, setDetailEvent] = useState(null);

  const onOpenDetail = useCallback((ev) => {
    setDetailEvent(ev);
    setDetailOpen(true);
  }, []);

  const onCloseDetail = useCallback(() => {
    setDetailEvent(null);
    setDetailOpen(false);
  }, []);

  return {
    detailOpen,
    detailEvent,
    onOpenDetail,
    onCloseDetail,
  };
}
