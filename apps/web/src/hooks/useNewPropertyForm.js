import { useState, useCallback, useEffect } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  parseIntegerFromInput,
  parseNumberFromInput,
} from "@/utils/formatters";
import { normalizePhoneToE164_TR as normalizePhoneToE164 } from "@/utils/phone";

export function useNewPropertyForm({ user, upload }) {
  const queryClient = useQueryClient();

  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  // Property info
  const [title, setTitle] = useState("");
  const [propertyStatus, setPropertyStatus] = useState("for_sale");

  const [currency, setCurrency] = useState("TRY");
  const [priceInput, setPriceInput] = useState("");
  const [depositInput, setDepositInput] = useState("");
  const [duesInput, setDuesInput] = useState("");

  const [addressLine, setAddressLine] = useState("");
  const [city, setCity] = useState("");
  const [postalCode, setPostalCode] = useState("");
  const [country, setCountry] = useState("");

  const [housingType, setHousingType] = useState("");
  const [housingShape, setHousingShape] = useState("");

  const [bedrooms, setBedrooms] = useState("");
  const [livingRooms, setLivingRooms] = useState("");
  const [bathrooms, setBathrooms] = useState("");

  const [grossAreaInput, setGrossAreaInput] = useState("");
  const [netAreaInput, setNetAreaInput] = useState("");

  const [totalFloors, setTotalFloors] = useState("");
  const [floorNumber, setFloorNumber] = useState("");
  const [buildingAge, setBuildingAge] = useState("");

  const [heatingType, setHeatingType] = useState("");
  const [elevator, setElevator] = useState("");
  const [parkingType, setParkingType] = useState("");

  const [titleDeedStatus, setTitleDeedStatus] = useState("");
  const [furnishedStatus, setFurnishedStatus] = useState("");
  const [mortgageEligible, setMortgageEligible] = useState("");
  const [constructionType, setConstructionType] = useState("");
  const [usageStatus, setUsageStatus] = useState("");
  const [facade, setFacade] = useState("");

  const [description, setDescription] = useState("");

  // Features
  const [featuresInterior, setFeaturesInterior] = useState([]);
  const [featuresExterior, setFeaturesExterior] = useState([]);

  // Owner info
  const [ownerMode, setOwnerMode] = useState("existing");
  const [ownerSearch, setOwnerSearch] = useState("");
  const [ownerDropdownOpen, setOwnerDropdownOpen] = useState(false);
  const [selectedOwnerClientId, setSelectedOwnerClientId] = useState(null);

  const [newOwnerName, setNewOwnerName] = useState("");
  const [newOwnerEmail, setNewOwnerEmail] = useState("");
  const [newOwnerPhone, setNewOwnerPhone] = useState("");
  const [newOwnerCountryCode, setNewOwnerCountryCode] = useState("Turkey");
  const [newOwnerCity, setNewOwnerCity] = useState("");

  // Photos
  const [photoFiles, setPhotoFiles] = useState([]);
  const [photoPreviews, setPhotoPreviews] = useState([]);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const urls = photoFiles.map((file) => URL.createObjectURL(file));
    setPhotoPreviews(urls);

    return () => {
      urls.forEach((u) => URL.revokeObjectURL(u));
    };
  }, [photoFiles]);

  const onPickPhotos = useCallback((e) => {
    const files = Array.from(e.target.files || []);
    setPhotoFiles(files);
  }, []);

  const resetOwnerSelection = useCallback(() => {
    setSelectedOwnerClientId(null);
    setOwnerSearch("");
    setOwnerDropdownOpen(false);
  }, []);

  const createClientMutation = useMutation({
    mutationFn: async (payload) => {
      const res = await fetch("/api/clients", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(
          `When posting /api/clients, the response was [${res.status}] ${res.statusText}`,
        );
      }

      return res.json();
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["clients", user?.id] });
    },
  });

  const onSubmit = useCallback(
    async (e) => {
      e.preventDefault();
      setError(null);
      setSubmitting(true);

      try {
        if (!title.trim()) {
          throw new Error("Please enter a property title");
        }

        let ownerClientId = null;

        if (ownerMode === "existing") {
          if (!selectedOwnerClientId) {
            // UX: open the dropdown so the user can immediately pick.
            setOwnerDropdownOpen(true);
            throw new Error(
              "Please select an owner client (or create a new one)",
            );
          }
          ownerClientId = selectedOwnerClientId;
        } else {
          if (!newOwnerName.trim()) {
            throw new Error("Please enter the owner's name");
          }

          const normalizedOwnerPhone = newOwnerPhone
            ? normalizePhoneToE164(newOwnerPhone)
            : null;

          const createdClient = await createClientMutation.mutateAsync({
            client_type: "owner",
            full_name: newOwnerName.trim(),
            email: newOwnerEmail || null,
            phone: normalizedOwnerPhone || null,
            notes: null,
            country: newOwnerCountryCode || null,
            city: newOwnerCity || null,
          });

          ownerClientId = createdClient.id;
        }

        // Upload photos first
        const uploadedPhotoUrls = [];
        if (photoFiles.length > 0) {
          for (const file of photoFiles) {
            const { url, error: uploadError } = await upload({ file });
            if (uploadError) {
              throw new Error(uploadError);
            }
            uploadedPhotoUrls.push(url);
          }
        }

        const parsedPrice = parseIntegerFromInput(priceInput);
        const parsedDeposit = parseIntegerFromInput(depositInput);
        const parsedDues = parseIntegerFromInput(duesInput);
        const parsedGross = parseNumberFromInput(grossAreaInput);
        const parsedNet = parseNumberFromInput(netAreaInput);

        const parsedBedrooms = bedrooms ? Number(bedrooms) : null;
        const parsedLiving = livingRooms ? Number(livingRooms) : null;

        const computedRooms =
          Number.isFinite(parsedBedrooms) && Number.isFinite(parsedLiving)
            ? parsedBedrooms + parsedLiving
            : null;

        const payload = {
          title: title.trim(),
          property_status: propertyStatus,
          address_line: addressLine || null,
          city: city || null,
          postal_code: postalCode || null,
          country: country || null,
          price: parsedPrice,
          currency: currency || null,
          size_sqm: parseNumberFromInput(grossAreaInput) || null,
          rooms: computedRooms,
          description: description || null,
          owner_client_id: ownerClientId,
          photos: uploadedPhotoUrls,

          housing_type: housingType || null,
          housing_shape: housingShape || null,
          bedrooms: bedrooms ? Number(bedrooms) : null,
          living_rooms: livingRooms ? Number(livingRooms) : null,
          bathrooms: bathrooms ? Number(bathrooms) : null,
          gross_area_sqm: parsedGross,
          net_area_sqm: parsedNet,
          total_floors: totalFloors ? Number(totalFloors) : null,
          floor_number: floorNumber ? Number(floorNumber) : null,
          building_age: buildingAge ? Number(buildingAge) : null,
          heating_type: heatingType || null,
          elevator:
            elevator === "yes" ? true : elevator === "no" ? false : null,
          parking_type: parkingType || null,
          title_deed_status: titleDeedStatus || null,
          furnished_status: furnishedStatus || null,
          mortgage_eligible:
            mortgageEligible === "yes"
              ? true
              : mortgageEligible === "no"
                ? false
                : null,
          construction_type: constructionType || null,
          usage_status: usageStatus || null,
          facade: facade || null,
          deposit: parsedDeposit,
          dues: parsedDues,
          // Always send arrays for these jsonb columns.
          features_interior: Array.isArray(featuresInterior)
            ? featuresInterior
            : [],
          features_exterior: Array.isArray(featuresExterior)
            ? featuresExterior
            : [],
          // Location features removed from the UI; keep backend-compatible by omitting it.
        };

        const res = await fetch("/api/properties", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!res.ok) {
          let details = null;
          try {
            details = await res.json();
          } catch {
            // ignore
          }

          const serverMessage =
            details?.message || details?.error || "Could not create property";

          throw new Error(
            `When posting /api/properties, the response was [${res.status}] ${res.statusText} — ${serverMessage}`,
          );
        }

        const createdProperty = await res.json();
        if (typeof window !== "undefined") {
          window.location.href = `/properties/${createdProperty.id}`;
        }
      } catch (err) {
        console.error(err);
        setError(err?.message || "Could not create property");
        setSubmitting(false);
      }
    },
    [
      addressLine,
      bathrooms,
      bedrooms,
      buildingAge,
      city,
      constructionType,
      country,
      createClientMutation,
      currency,
      depositInput,
      description,
      duesInput,
      elevator,
      facade,
      featuresExterior,
      featuresInterior,
      floorNumber,
      grossAreaInput,
      heatingType,
      housingShape,
      housingType,
      livingRooms,
      mortgageEligible,
      newOwnerCity,
      newOwnerCountryCode,
      newOwnerEmail,
      newOwnerName,
      newOwnerPhone,
      ownerMode,
      parkingType,
      photoFiles,
      postalCode,
      priceInput,
      propertyStatus,
      selectedOwnerClientId,
      title,
      titleDeedStatus,
      totalFloors,
      upload,
      usageStatus,
      user?.id,
      setOwnerDropdownOpen,
    ],
  );

  return {
    error,
    submitting,
    title,
    setTitle,
    propertyStatus,
    setPropertyStatus,
    currency,
    setCurrency,
    priceInput,
    setPriceInput,
    depositInput,
    setDepositInput,
    duesInput,
    setDuesInput,
    addressLine,
    setAddressLine,
    city,
    setCity,
    postalCode,
    setPostalCode,
    country,
    setCountry,
    housingType,
    setHousingType,
    housingShape,
    setHousingShape,
    bedrooms,
    setBedrooms,
    livingRooms,
    setLivingRooms,
    bathrooms,
    setBathrooms,
    grossAreaInput,
    setGrossAreaInput,
    netAreaInput,
    setNetAreaInput,
    totalFloors,
    setTotalFloors,
    floorNumber,
    setFloorNumber,
    buildingAge,
    setBuildingAge,
    heatingType,
    setHeatingType,
    elevator,
    setElevator,
    parkingType,
    setParkingType,
    titleDeedStatus,
    setTitleDeedStatus,
    furnishedStatus,
    setFurnishedStatus,
    mortgageEligible,
    setMortgageEligible,
    constructionType,
    setConstructionType,
    usageStatus,
    setUsageStatus,
    facade,
    setFacade,
    description,
    setDescription,
    featuresInterior,
    setFeaturesInterior,
    featuresExterior,
    setFeaturesExterior,
    ownerMode,
    setOwnerMode,
    ownerSearch,
    setOwnerSearch,
    ownerDropdownOpen,
    setOwnerDropdownOpen,
    selectedOwnerClientId,
    setSelectedOwnerClientId,
    newOwnerName,
    setNewOwnerName,
    newOwnerEmail,
    setNewOwnerEmail,
    newOwnerPhone,
    setNewOwnerPhone,
    newOwnerCountryCode,
    setNewOwnerCountryCode,
    newOwnerCity,
    setNewOwnerCity,
    photoFiles,
    setPhotoFiles,
    photoPreviews,
    setPhotoPreviews,
    onPickPhotos,
    resetOwnerSelection,
    onSubmit,
  };
}
