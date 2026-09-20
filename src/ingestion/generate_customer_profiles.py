import random

import numpy as np
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# =========================================================
# LOAD CUSTOMERS
# =========================================================

customers = pd.read_csv(
    "data/synthetic/customers.csv"
)


# =========================================================
# ACTIVITY PROFILES
# =========================================================

ACTIVITY_PROFILES = {
    "LOW": 0.50,
    "MEDIUM": 0.40,
    "HIGH": 0.10,
}


# =========================================================
# CURRENCY BY COUNTRY
# =========================================================

CURRENCY_BY_COUNTRY = {
    "Nigeria": "NGN",
    "Ghana": "GHS",
}


# =========================================================
# GENERATE CUSTOMER PROFILES
# =========================================================

def generate_customer_profiles():

    profiles = []

    for _, customer in customers.iterrows():

        customer_id = customer["customer_id"]
        country = customer["country"]

        currency = CURRENCY_BY_COUNTRY[
            country
        ]

        # ---------------------------------------------
        # Activity profile
        # ---------------------------------------------

        activity_profile = random.choices(
            list(ACTIVITY_PROFILES.keys()),
            weights=list(
                ACTIVITY_PROFILES.values()
            ),
            k=1,
        )[0]


        # ---------------------------------------------
        # Expected transactions per day
        # ---------------------------------------------

        if activity_profile == "LOW":

            expected_daily_transactions = np.random.uniform(
                0.2,
                1.0,
            )

        elif activity_profile == "MEDIUM":

            expected_daily_transactions = np.random.uniform(
                1.0,
                4.0,
            )

        else:

            expected_daily_transactions = np.random.uniform(
                4.0,
                12.0,
            )


        # ---------------------------------------------
        # Typical transaction amount
        # ---------------------------------------------
        #
        # Amounts are generated separately for each
        # currency so NGN and GHS have different scales.
        #
        # These are synthetic assumptions for FinGuard.
        # ---------------------------------------------

        if currency == "NGN":

            if activity_profile == "LOW":

                typical_amount = np.random.lognormal(
                    mean=9.0,
                    sigma=0.6,
                )

            elif activity_profile == "MEDIUM":

                typical_amount = np.random.lognormal(
                    mean=10.0,
                    sigma=0.7,
                )

            else:

                typical_amount = np.random.lognormal(
                    mean=11.0,
                    sigma=0.8,
                )

        else:

            if activity_profile == "LOW":

                typical_amount = np.random.lognormal(
                    mean=3.5,
                    sigma=0.6,
                )

            elif activity_profile == "MEDIUM":

                typical_amount = np.random.lognormal(
                    mean=4.5,
                    sigma=0.7,
                )

            else:

                typical_amount = np.random.lognormal(
                    mean=5.5,
                    sigma=0.8,
                )


        profiles.append({

            "customer_id":
                customer_id,

            "activity_profile":
                activity_profile,

            "expected_daily_transactions":
                round(
                    expected_daily_transactions,
                    2,
                ),

            "typical_transaction_amount":
                round(
                    typical_amount,
                    2,
                ),

            "currency":
                currency,
        })


    return pd.DataFrame(
        profiles
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    profiles = (
        generate_customer_profiles()
    )

    print(
        profiles.head()
    )

    print()

    print(
        f"Number of customer profiles: "
        f"{len(profiles):,}"
    )

    print()

    print(
        "Activity profile distribution:"
    )

    print(
        profiles[
            "activity_profile"
        ].value_counts()
    )

    print()

    print(
        "Currency distribution:"
    )

    print(
        profiles[
            "currency"
        ].value_counts()
    )

    profiles.to_csv(
        "data/synthetic/customer_profiles.csv",
        index=False,
    )

    print()

    print(
        "Saved to "
        "data/synthetic/customer_profiles.csv"
    )