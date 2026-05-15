import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

export function useContractDetail(contractId, userId) {
  const queryClient = useQueryClient();

  const {
    data: contract,
    isLoading,
    error,
  } = useQuery({
    queryKey: ["contract", userId, contractId],
    queryFn: async () => {
      const res = await fetch(`/api/contracts/${contractId}`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to load contract");
      }
      return res.json();
    },
    enabled: !!userId && !!contractId,
  });

  const invalidateQueries = async () => {
    await queryClient.invalidateQueries({
      queryKey: ["contract", userId, contractId],
    });
    await queryClient.invalidateQueries({
      queryKey: ["contracts", userId],
    });
    if (contract?.property_id) {
      await queryClient.invalidateQueries({
        queryKey: ["property", userId, contract.property_id],
      });
    }
  };

  const markSignedMutation = useMutation({
    mutationFn: async ({ agentName, clientName }) => {
      const res = await fetch(`/api/contracts/${contractId}/mark-signed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          signed_by_agent_name: agentName,
          signed_by_client_name: clientName,
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to mark as signed");
      }

      return res.json();
    },
    onSuccess: invalidateQueries,
  });

  const markUnsignedMutation = useMutation({
    mutationFn: async () => {
      const res = await fetch(`/api/contracts/${contractId}/mark-unsigned`, {
        method: "POST",
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to mark as unsigned");
      }

      return res.json();
    },
    onSuccess: invalidateQueries,
  });

  const regeneratePdfMutation = useMutation({
    mutationFn: async () => {
      const res = await fetch(`/api/contracts/${contractId}/regenerate-pdf`, {
        method: "POST",
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to regenerate PDF");
      }

      return res.json();
    },
    onSuccess: invalidateQueries,
  });

  const updateFieldsMutation = useMutation({
    mutationFn: async (nextFields) => {
      const res = await fetch(`/api/contracts/${contractId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filledFields: nextFields }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Failed to update contract");
      }

      return res.json();
    },
    onSuccess: invalidateQueries,
  });

  return {
    contract,
    isLoading,
    error,
    markSignedMutation,
    markUnsignedMutation,
    regeneratePdfMutation,
    updateFieldsMutation,
  };
}
