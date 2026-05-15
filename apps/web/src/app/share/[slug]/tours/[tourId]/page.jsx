import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import Fake360Viewer from "@/components/Fake360Viewer";
import Splat3DViewer from "@/components/Splat3DViewer";
import { StatusBanner } from "@/components/StatusBanner";

export default function SharedTourPage(props) {
  const slug = props?.params?.slug;
  const tourId = props?.params?.tourId;

  const { data, isLoading, error } = useQuery({
    queryKey: ["share", slug],
    queryFn: async () => {
      const res = await fetch(`/api/share/${slug}`);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        const err = new Error(body?.error || "Not found");
        err.status = res.status;
        throw err;
      }
      return res.json();
    },
    enabled: !!slug,
    // Prevent background refetches from creating extra access log entries.
    refetchOnWindowFocus: false,
  });

  const tour = useMemo(() => {
    const tours = Array.isArray(data?.property?.tours)
      ? data.property.tours
      : [];
    return tours.find((t) => String(t?.id) === String(tourId)) || null;
  }, [data?.property?.tours, tourId]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-5xl mx-auto">
          <p className="text-gray-600 font-jetbrains-mono">Loading...</p>
        </div>
      </div>
    );
  }

  if (error) {
    const status = Number(error?.status || 0);
    const message = String(error?.message || "");

    const isExpired =
      status === 410 || message.toLowerCase().includes("expired");

    const title = isExpired ? "Link expired" : "Not available";
    const text = isExpired
      ? "This share link has expired."
      : "This item is no longer available.";

    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-5xl mx-auto">
          <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
            <StatusBanner variant="info" title={title}>
              {text}
            </StatusBanner>
          </div>
        </div>
      </div>
    );
  }

  if (!tour || !tour?.tour_payload) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-5xl mx-auto">
          <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200">
            <StatusBanner variant="info" title="Not available">
              This item is no longer available.
            </StatusBanner>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-5xl mx-auto">
        <div className="bg-white rounded-xl p-4 sm:p-6 shadow-sm border border-gray-200">
          {tour?.tour_type === "splat3d" ||
          tour?.tour_payload?.type === "splat3d" ? (
            <Splat3DViewer tourPayload={tour.tour_payload} height={620} />
          ) : (
            <Fake360Viewer tourPayload={tour.tour_payload} />
          )}
        </div>
      </div>
    </div>
  );
}
