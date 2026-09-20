import random
from datetime import timedelta

import pandas as pd
from faker import Faker


SEED = 42
random.seed(SEED)
Faker.seed(SEED)

fake = Faker()

SIMULATION_END_DATE = pd.Timestamp("2026-08-31")


COUNTRIES = {
    "Nigeria": {
        "currency": "NGN",
        "cities": [
            "Lagos",
            "Abuja",
            "Port Harcourt",
            "Ibadan",
            "Kano",
            "Benin City",
            "Enugu",
        ],
    },
    "Ghana": {
        "currency": "GHS",
        "cities": [
            "Accra",
            "Kumasi",
            "Tema",
            "Tamale",
            "Takoradi",
        ],
    },
}


COUNTRY_WEIGHTS = {
    "Nigeria": 0.75,
    "Ghana": 0.25,
}


MERCHANT_CATEGORIES = {
    "Retail": 0.20,
    "Food & Restaurants": 0.18,
    "Electronics": 0.10,
    "E-commerce": 0.12,
    "Telecommunications": 0.10,
    "Travel & Transport": 0.07,
    "Utilities": 0.08,
    "Professional Services": 0.05,
    "Entertainment": 0.05,
    "Healthcare": 0.05,
}


MERCHANT_TYPES = {
    "Physical": 0.55,
    "Online": 0.30,
    "Marketplace": 0.15,
}


RISK_PROFILES = {
    "LOW": 0.70,
    "MEDIUM": 0.23,
    "HIGH": 0.07,
}
def generate_merchant_id(index):
    return f"MERCH_{index:06d}"
def generate_merchants(num_merchants=2_000):

    merchants = []

    for i in range(1, num_merchants + 1):

        country = random.choices(
            list(COUNTRIES.keys()),
            weights=list(COUNTRY_WEIGHTS.values()),
            k=1,
        )[0]

        city = random.choice(
            COUNTRIES[country]["cities"]
        )

        currency = COUNTRIES[country]["currency"]

        category = random.choices(
            list(MERCHANT_CATEGORIES.keys()),
            weights=list(MERCHANT_CATEGORIES.values()),
            k=1,
        )[0]

        merchant_type = random.choices(
            list(MERCHANT_TYPES.keys()),
            weights=list(MERCHANT_TYPES.values()),
            k=1,
        )[0]

        risk_profile = random.choices(
            list(RISK_PROFILES.keys()),
            weights=list(RISK_PROFILES.values()),
            k=1,
        )[0]

        merchant_age_days = random.randint(
            1,
            1825,
        )

        account_created_at = (
            SIMULATION_END_DATE
            - timedelta(days=merchant_age_days)
        )

        merchants.append({
            "merchant_id": generate_merchant_id(i),
            "merchant_name": fake.company(),
            "country": country,
            "city": city,
            "merchant_category": category,
            "merchant_type": merchant_type,
            "currency": currency,
            "account_created_at": account_created_at,
            "risk_profile": risk_profile,
        })

    return pd.DataFrame(merchants)
if __name__ == "__main__":

    merchants = generate_merchants()

    print(merchants.head())

    print()
    print(f"Number of merchants: {len(merchants):,}")

    merchants.to_csv(
        "data/synthetic/merchants.csv",
        index=False,
    )

    print()
    print("Saved to data/synthetic/merchants.csv")