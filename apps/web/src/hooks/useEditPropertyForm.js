import { useCallback, useEffect, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  parseIntegerFromInput,
  parseNumberFromInput,
} from "@/utils/formatters";
import { normalizePhoneToE164_TR as normalizePhoneToE164 } from "@/utils/phone";

export function useEditPropertyForm({ user, propertyId, property }) {
  const queryClient = useQueryClient();

  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const initializedRef = useRef(false);

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

  useEffect(() => {
    if (initializedRef.current) return;
    if (!property) return;

    initializedRef.current = true;

    setTitle(property.title || "");
    setPropertyStatus(property.property_status || "for_sale");

    setCurrency(property.currency || "TRY");
    setPriceInput(
      property.price !== null && property.price !== undefined
        ? String(property.price)
        : "",
    );
    setDepositInput(
      property.deposit !== null && property.deposit !== undefined
        ? String(property.deposit)
        : "",
    );
    setDuesInput(
      property.dues !== null && property.dues !== undefined
        ? String(property.dues)
        : "",
    );

    setAddressLine(property.address_line || "");
    setCity(property.city || "");
    setPostalCode(property.postal_code || "");
    setCountry(property.country || "");

    setHousingType(property.housing_type || "");
    setHousingShape(property.housing_shape || "");

    setBedrooms(
      property.bedrooms !== null && property.bedrooms !== undefined
        ? String(property.bedrooms)
        : "",
    );
    setLivingRooms(
      property.living_rooms !== null && property.living_rooms !== undefined
        ? String(property.living_rooms)
        : "",
    );
    setBathrooms(
      property.bathrooms !== null && property.bathrooms !== undefined
        ? String(property.bathrooms)
        : "",
    );

    setGrossAreaInput(
      property.gross_area_sqm !== null && property.gross_area_sqm !== undefined
        ? String(property.gross_area_sqm)
        : "",
    );
    setNetAreaInput(
      property.net_area_sqm !== null && property.net_area_sqm !== undefined
        ? String(property.net_area_sqm)
        : "",
    );

    setTotalFloors(
      property.total_floors !== null && property.total_floors !== undefined
        ? String(property.total_floors)
        : "",
    );
    setFloorNumber(
      property.floor_number !== null && property.floor_number !== undefined
        ? String(property.floor_number)
        : "",
    );
    setBuildingAge(
      property.building_age !== null && property.building_age !== undefined
        ? String(property.building_age)
        : "",
    );

    setHeatingType(property.heating_type || "");
    setElevator(
      property.elevator === true
        ? "yes"
        : property.elevator === false
          ? "no"
          : "",
    );
    setParkingType(property.parking_type || "");

    setTitleDeedStatus(property.title_deed_status || "");
    setFurnishedStatus(property.furnished_status || "");
    setMortgageEligible(
      property.mortgage_eligible === true
        ? "yes"
        : property.mortgage_eligible === false
          ? "no"
          : "",
    );
    setConstructionType(property.construction_type || "");
    setUsageStatus(property.usage_status || "");
    setFacade(property.facade || "");

    setDescription(property.description || "");

    setFeaturesInterior(
      Array.isArray(property.features_interior)
        ? property.features_interior
        : [],
    );
    setFeaturesExterior(
      Array.isArray(property.features_exterior)
        ? property.features_exterior
        : [],
    );

    // Owner (existing)
    if (property.owner_client_id) {
      setOwnerMode("existing");
      setSelectedOwnerClientId(property.owner_client_id);
      setOwnerSearch(property.owner_name || "");
    }
  }, [property]);

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
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || "Could not create client");
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
        if (!propertyId) {
          throw new Error("Missing property ID");
        }

        if (!title.trim()) {
          throw new Error("Please enter a property title");
        }

        let ownerClientId = null;

        if (ownerMode === "existing") {
          if (!selectedOwnerClientId) {
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
          size_sqm: parsedGross,
          rooms: computedRooms,
          description: description || null,
          owner_client_id: ownerClientId,

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
          features_interior: Array.isArray(featuresInterior)
            ? featuresInterior
            : [],
          features_exterior: Array.isArray(featuresExterior)
            ? featuresExterior
            : [],
        };

        const res = await fetch(`/api/properties/${propertyId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          throw new Error(body?.error || "Could not update property");
        }

        const updated = await res.json();
        await queryClient.invalidateQueries({
          queryKey: ["property", user?.id, propertyId],
        });

        if (typeof window !== "undefined") {
          window.location.href = `/properties/${updated.id}`;
        }
      } catch (err) {
        console.error(err);
        setError(err?.message || "Could not update property");
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
      postalCode,
      priceInput,
      propertyId,
      propertyStatus,
      selectedOwnerClientId,
      title,
      titleDeedStatus,
      totalFloors,
      usageStatus,
      user?.id,
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

    resetOwnerSelection,
    onSubmit,
  };
}
