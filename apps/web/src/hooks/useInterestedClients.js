import { useCallback, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";

export function useInterestedClients(
  userId,
  propertyId,
  interestedClients,
  refetchProperty,
) {
  const [interestedModalOpen, setInterestedModalOpen] = useState(false);
  const [interestedSearch, setInterestedSearch] = useState("");
  const [selectedInterestedIds, setSelectedInterestedIds] = useState([]);

  const { data: allClients = [], isLoading: clientsLoading } = useQuery({
    queryKey: ["clients", userId],
    queryFn: async () => {
      const res = await fetch("/api/clients");
      if (!res.ok) {
        throw new Error(
          `When fetching /api/clients, the response was [${res.status}] ${res.statusText}`,
        );
      }
      return res.json();
    },
    enabled: !!userId && interestedModalOpen,
  });

  const filteredClients = useMemo(() => {
    const normalized = interestedSearch.trim().toLowerCase();
    const list = allClients.slice().sort((a, b) => {
      const aa = (a.full_name || "").toLowerCase();
      const bb = (b.full_name || "").toLowerCase();
      return aa.localeCompare(bb);
    });

    if (!normalized) return list;

    return list.filter((c) => {
      const name = (c.full_name || "").toLowerCase();
      const email = (c.email || "").toLowerCase();
      const phone = (c.phone || "").toLowerCase();
      return (
        name.includes(normalized) ||
        email.includes(normalized) ||
        phone.includes(normalized)
      );
    });
  }, [allClients, interestedSearch]);

  const openInterestedModal = useCallback(() => {
    const currentIds = interestedClients.map((c) => c.id);
    setSelectedInterestedIds(currentIds);
    setInterestedSearch("");
    setInterestedModalOpen(true);
  }, [interestedClients]);

  const saveInterestedMutation = useMutation({
    mutationFn: async (clientIds) => {
      const res = await fetch(
        `/api/properties/${propertyId}/interested-clients`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ clientIds }),
        },
      );
      if (!res.ok) {
        throw new Error(
          `When putting /api/properties/${propertyId}/interested-clients, the response was [${res.status}] ${res.statusText}`,
        );
      }
      return res.json();
    },
    onSuccess: async () => {
      await refetchProperty();
      setInterestedModalOpen(false);
    },
  });

  const removeInterestedClient = useCallback(
    (clientId) => {
      const currentIds = interestedClients.map((c) => c.id);
      const next = currentIds.filter((id) => id !== clientId);
      setSelectedInterestedIds(next);
      saveInterestedMutation.mutate(next);
    },
    [interestedClients, saveInterestedMutation],
  );

  return {
    interestedModalOpen,
    setInterestedModalOpen,
    interestedSearch,
    setInterestedSearch,
    selectedInterestedIds,
    setSelectedInterestedIds,
    allClients,
    clientsLoading,
    filteredClients,
    openInterestedModal,
    saveInterestedMutation,
    removeInterestedClient,
  };
}
