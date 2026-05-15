export const STATUS_OPTIONS = [
  { value: "for_sale", label: "For Sale" },
  { value: "for_rent", label: "For Rent" },
];

export const CURRENCY_OPTIONS = [
  { value: "TRY", label: "TRY" },
  { value: "USD", label: "USD" },
  { value: "EUR", label: "EUR" },
  { value: "GBP", label: "GBP" },
];

export const HOUSING_TYPE_OPTIONS = [
  { value: "Apartment", label: "Apartment" },
  { value: "Residence", label: "Residence" },
  { value: "Villa", label: "Villa" },
  { value: "Detached", label: "Detached House" },
  { value: "Office", label: "Office" },
  { value: "Land", label: "Land" },
];

export const HOUSING_SHAPE_OPTIONS = [
  // Layout options (English)
  { value: "studio", label: "Studio" },
  { value: "1+1", label: "1+1" },
  { value: "2+1", label: "2+1" },
  { value: "3+1", label: "3+1" },
  { value: "4+1", label: "4+1" },
  { value: "5+1", label: "5+1" },
  { value: "6+1", label: "6+1" },
  { value: "2+2_plus", label: "2+2 and above" },
  { value: "3+2", label: "3+2" },
  { value: "3+3_plus", label: "3+3 and above" },
  { value: "4+2", label: "4+2" },
  { value: "4+3", label: "4+3" },
  { value: "4+4_plus", label: "4+4 and above" },
  { value: "5+2", label: "5+2" },
  { value: "5+3", label: "5+3" },
  { value: "5+4_plus", label: "5+4 and above" },
  { value: "6", label: "6" },
  { value: "6+2", label: "6+2" },
  { value: "6+3", label: "6+3" },
  { value: "6+4_plus", label: "6+4 and above" },
  { value: "7+2", label: "7+2" },
  { value: "7+3", label: "7+3" },
  { value: "7+4_plus", label: "7+4 and above" },
  { value: "8+2", label: "8+2" },
  { value: "8+3", label: "8+3" },
  { value: "8+4_plus", label: "8+4 and above" },
  { value: "9+2", label: "9+2" },
  { value: "9+3", label: "9+3" },
  { value: "9+4_plus", label: "9+4 and above" },
];

export const HEATING_TYPE_OPTIONS = [
  { value: "Natural Gas", label: "Natural Gas" },
  { value: "Central", label: "Central" },
  { value: "Air Conditioner", label: "Air Conditioner" },
  { value: "Stove", label: "Stove" },
  { value: "Floor Heating", label: "Floor Heating" },
  { value: "Other", label: "Other" },
];

export const PARKING_OPTIONS = [
  { value: "None", label: "None" },
  { value: "Open", label: "Open" },
  { value: "Closed", label: "Closed" },
];

export const TITLE_DEED_OPTIONS = [
  { value: "Condominium", label: "Condominium" },
  { value: "Shared", label: "Shared" },
  { value: "Land", label: "Land" },
  { value: "Other", label: "Other" },
];

export const FURNISHED_OPTIONS = [
  { value: "Unfurnished", label: "Unfurnished" },
  { value: "Partly", label: "Partly furnished" },
  { value: "Furnished", label: "Furnished" },
];

export const CONSTRUCTION_OPTIONS = [
  { value: "Reinforced Concrete", label: "Reinforced Concrete" },
  { value: "Steel", label: "Steel" },
  { value: "Wood", label: "Wood" },
  { value: "Other", label: "Other" },
];

export const USAGE_STATUS_OPTIONS = [
  { value: "Empty", label: "Empty" },
  { value: "Tenant", label: "Tenant" },
  { value: "Owner", label: "Owner" },
  { value: "Other", label: "Other" },
];

export const FACADE_OPTIONS = [
  { value: "North", label: "North" },
  { value: "South", label: "South" },
  { value: "East", label: "East" },
  { value: "West", label: "West" },
  { value: "NorthEast", label: "North-East" },
  { value: "NorthWest", label: "North-West" },
  { value: "SouthEast", label: "South-East" },
  { value: "SouthWest", label: "South-West" },
];

// Features (grouped)
export const FEATURES_INTERIOR_GROUPS = [
  {
    title: "Connectivity & Technology",
    options: [
      "Fiber Internet",
      "In-unit WiFi",
      "Cable TV / Satellite",
      "Video Intercom",
      "Alarm System",
    ],
  },
  {
    title: "Kitchen",
    options: ["Built-in Kitchen", "Appliances Included", "Furnished"],
  },
  {
    title: "Flooring & Structure",
    options: [
      "Laminate Flooring",
      "Engineered Wood Flooring",
      "Ceramic Flooring",
      "Marble Flooring",
      "Double Glazing",
    ],
  },
  {
    title: "Heating & Utilities",
    options: [
      "Underfloor Heating",
      "Natural Gas Infrastructure",
      "Water Heater",
      "Air Conditioning",
    ],
  },
  {
    title: "Space & Comfort",
    options: [
      "Balcony",
      "Terrace",
      "Pantry / Storage Room",
      "En-suite Bathroom",
      "Hilton Bathroom",
      "Dressing Room",
      "Laundry Room",
      "Built-in Wardrobe",
      "Coat Closet",
    ],
  },
  {
    title: "Luxury & Premium Features",
    options: [
      "Sauna",
      "Finnish Bath",
      "Turkish Bath",
      "Indoor Swimming Pool (In-unit)",
      "Home Cinema Room",
      "Fitness / Gym Room",
      "Hobby Room",
      "Wine Room / Cellar",
      "Game Room",
      "Music Room",
      "Smart Home System",
    ],
  },
  {
    title: "Extra",
    options: ["Shutters", "Electric Shutters", "Barbecue"],
  },
];

export const FEATURES_EXTERIOR_GROUPS = [
  {
    title: "Community & Security",
    options: [
      "Gated Community",
      "Security",
      "CCTV System",
      "Concierge",
      "Fire Escape",
    ],
  },
  {
    title: "Parking & Transport",
    options: ["Outdoor Parking", "EV Charging Station"],
  },
  {
    title: "Social Areas & Sports",
    options: [
      "Outdoor Swimming Pool",
      "Indoor Swimming Pool",
      "Fitness / Gym",
      "Sauna",
      "Turkish Bath",
      "Tennis Court",
      "Playground",
    ],
  },
  {
    title: "Garden & Views",
    options: ["Garden", "Garden View", "City View"],
  },
  {
    title: "Technical & Infrastructure",
    options: [
      "Elevator",
      "Thermal Insulation",
      "Generator",
      "Hydrophore",
      "Water Tank",
    ],
  },
  {
    title: "Accessibility",
    options: [
      "Accessible Entrance",
      "Wheelchair Accessible Elevator",
      "Accessible Parking",
      "Step-Free Entrance",
    ],
  },
];

// NOTE: Location features removed from the form UI per product requirements.
// The database column (features_location) still exists for backward compatibility.
