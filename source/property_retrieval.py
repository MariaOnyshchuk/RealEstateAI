
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Property(BaseModel):
    # Core identifiers
    property_url: str = ""
    full_street_line: Optional[str] = None
    street: Optional[str] = None
    unit: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[int] = None
    county: Optional[str] = None

    # Descriptive features
    text: Optional[str] = None
    style: Optional[str] = None
    status: Optional[str] = None
    neighborhoods: Optional[Any] = None  # could be list or string depending on DB

    # Structure / physical features
    beds: Optional[float] = None
    full_baths: Optional[float] = None
    half_baths: Optional[float] = None
    sqft: Optional[float] = None
    lot_sqft: Optional[float] = None
    stories: Optional[float] = None
    year_built: Optional[float] = None
    parking_garage: Optional[float] = None

    # Pricing
    list_price: Optional[float] = None
    sold_price: Optional[float] = None
    last_sold_price: Optional[float] = None
    assessed_value: Optional[float] = None
    estimated_value: Optional[float] = None
    price_per_sqft: Optional[float] = None

    # Dates
    list_date: Optional[str] = None
    pending_date: Optional[str] = None
    last_sold_date: Optional[str] = None

    # Financial / investment
    avg_monthly_rent: Optional[float] = None
    annual_rent: Optional[float] = None
    gross_rental_yield: Optional[float] = None
    net_rental_yield: Optional[float] = None
    roi: Optional[float] = None
    avg_annual_insurance: Optional[float] = None
    maintenance: Optional[float] = None
    annual_cost: Optional[float] = None
    tax: Optional[float] = None
    tax_history: Optional[Any] = None

    # Geographic
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # Amenities counts (each is int)
    cafe: Optional[int] = None
    restaurant: Optional[int] = None
    pharmacy: Optional[int] = None
    gym: Optional[int] = None
    library: Optional[int] = None
    museum: Optional[int] = None
    night_club: Optional[int] = None
    park: Optional[int] = None
    school: Optional[int] = None
    shopping_mall: Optional[int] = None
    stadium: Optional[int] = None
    supermarket: Optional[int] = None
    university: Optional[int] = None
    town_square: Optional[int] = None

    # Misc
    days_on_mls: Optional[int] = None
    crime_rate: Optional[float] = None
    target_group: Optional[str] = None
    new_construction: Optional[int] = None

    # Agent
    agent_name: Optional[str] = None
    agent_email: Optional[str] = None
    agent_phones: Optional[str] = None
    agent_mls_set: Optional[str] = None

    # Extra user field you had
    m: Optional[float] = None

    class Config:
        extra = "ignore"
        coerce_numbers_to_str = False
        arbitrary_types_allowed = True


class QueryResult(BaseModel):
    summary: str
    top_properties: List[Property] = Field(default_factory=list)
