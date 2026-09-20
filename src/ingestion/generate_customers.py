import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker
SIMULATION_END_DATE = datetime(2026, 8, 31)
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

fake = Faker()
Faker.seed(SEED)
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
def generate_customer_id(index):
    return f"CUST_{index:06d}"
def generate_customers(num_customers=10_000):

    customers = []

    country_names = list(COUNTRY_WEIGHTS.keys())
    country_probabilities = list(COUNTRY_WEIGHTS.values())

    for i in range(1, num_customers + 1):

        country = np.random.choice(
            country_names,
            p=country_probabilities
        )

        city = random.choice(COUNTRIES[country]["cities"])

        age = int(
            np.clip(
                np.random.normal(loc=34, scale=10),
                18,
                75
            )
        )

        account_age_days = random.randint(1, 1825)

        account_created_at = (
    SIMULATION_END_DATE - timedelta(days=account_age_days)
)

        customer = {
            "customer_id": generate_customer_id(i),
            "country": country,
            "city": city,
            "age": age,
            "account_created_at": account_created_at,
            "kyc_level": random.choice(
                ["Tier 1", "Tier 2", "Tier 3"]
            ),
            "account_status": random.choices(
                ["ACTIVE", "SUSPENDED", "CLOSED"],
                weights=[0.96, 0.025, 0.015],
                k=1,
            )[0],
        }

        customers.append(customer)

    return pd.DataFrame(customers)
if __name__ == "__main__":

    df_customers = generate_customers()

    print(df_customers.head())
    print()
    print(df_customers.shape)

    df_customers.to_csv(
        "data/synthetic/customers.csv",
        index=False
    )