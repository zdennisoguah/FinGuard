import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 8, 31)

CURRENCY_BY_COUNTRY = {
    "Nigeria": "NGN",
    "Ghana": "GHS",
}


# ============================================================
# TRANSACTION TYPES
# ============================================================

TRANSACTION_TYPES = [
    "P2P_TRANSFER",
    "MERCHANT_PAYMENT",
    "CARD_PAYMENT",
    "CASH_WITHDRAWAL",
]

TRANSACTION_TYPE_WEIGHTS = [
    0.35,
    0.35,
    0.20,
    0.10,
]


# ============================================================
# CHANNELS BY TRANSACTION TYPE
# ============================================================

CHANNELS_BY_TYPE = {
    "P2P_TRANSFER": {
        "MOBILE_APP": 0.60,
        "WEB": 0.15,
        "USSD": 0.25,
    },

    "MERCHANT_PAYMENT": {
        "MOBILE_APP": 0.40,
        "WEB": 0.15,
        "POS": 0.45,
    },

    "CARD_PAYMENT": {
        "MOBILE_APP": 0.20,
        "WEB": 0.35,
        "POS": 0.45,
    },

    "CASH_WITHDRAWAL": {
        "ATM": 0.85,
        "POS": 0.15,
    },
}


# ============================================================
# TRANSACTION STATUS
# ============================================================

STATUSES = [
    "SUCCESS",
    "FAILED",
    "REVERSED",
]

STATUS_WEIGHTS = [
    0.94,
    0.05,
    0.01,
]


# ============================================================
# LOAD DATA
# ============================================================

print("Loading source data...")

customers = pd.read_csv(
    "data/synthetic/customers.csv"
)

merchants = pd.read_csv(
    "data/synthetic/merchants.csv"
)

customer_devices = pd.read_csv(
    "data/synthetic/customer_devices.csv"
)

customer_profiles = pd.read_csv(
    "data/synthetic/customer_profiles.csv"
)


# ============================================================
# PREPARE DATA
# ============================================================

customers["account_created_at"] = pd.to_datetime(
    customers["account_created_at"]
)

customer_profiles["expected_daily_transactions"] = (
    customer_profiles["expected_daily_transactions"]
    .astype(float)
)


# ============================================================
# CREATE LOOKUPS
# ============================================================

customer_device_map = (
    customer_devices
    .groupby("customer_id")["device_id"]
    .apply(list)
    .to_dict()
)


merchant_map = (
    merchants
    .groupby("country")["merchant_id"]
    .apply(list)
    .to_dict()
)


customer_profile_map = (
    customer_profiles
    .set_index("customer_id")
    .to_dict("index")
)


customer_map = (
    customers
    .set_index("customer_id")
    .to_dict("index")
)


# ============================================================
# GENERATE RANDOM TRANSACTION TIME
# ============================================================

def generate_transaction_time(
    transaction_date
):
    """
    Generate a random timestamp during a given day.
    """

    seconds_in_day = 24 * 60 * 60 - 1

    random_seconds = random.randint(
        0,
        seconds_in_day
    )

    return transaction_date + timedelta(
        seconds=random_seconds
    )


# ============================================================
# GENERATE TRANSACTION AMOUNT
# ============================================================

def generate_amount(
    transaction_type,
    typical_amount
):
    """
    Generate a transaction amount around
    the customer's typical transaction amount.
    """

    if transaction_type == "CASH_WITHDRAWAL":
        sigma = 0.45

    elif transaction_type == "CARD_PAYMENT":
        sigma = 0.35

    elif transaction_type == "MERCHANT_PAYMENT":
        sigma = 0.40

    else:
        # P2P_TRANSFER
        sigma = 0.50

    variation = np.random.lognormal(
        mean=0.0,
        sigma=sigma
    )

    amount = typical_amount * variation

    return round(
        max(amount, 100),
        2
    )


# ============================================================
# SELECT CHANNEL
# ============================================================

def select_channel(
    transaction_type
):
    channels = list(
        CHANNELS_BY_TYPE[
            transaction_type
        ].keys()
    )

    probabilities = list(
        CHANNELS_BY_TYPE[
            transaction_type
        ].values()
    )

    return random.choices(
        channels,
        weights=probabilities,
        k=1
    )[0]


# ============================================================
# GENERATE SINGLE TRANSACTION
# ============================================================

def generate_transaction(
    transaction_id,
    customer_id,
    transaction_date
):
    """
    Generate one transaction for a customer.
    """

    customer = customer_map[
        customer_id
    ]

    profile = customer_profile_map[
        customer_id
    ]

    country = customer["country"]
    city = customer["city"]

    currency = profile["currency"]

    # --------------------------------------------------------
    # Transaction type
    # --------------------------------------------------------

    transaction_type = random.choices(
        TRANSACTION_TYPES,
        weights=TRANSACTION_TYPE_WEIGHTS,
        k=1
    )[0]

    # --------------------------------------------------------
    # Channel
    # --------------------------------------------------------

    channel = select_channel(
        transaction_type
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    available_devices = customer_device_map.get(
        customer_id,
        []
    )

    if available_devices:
        device_id = random.choice(
            available_devices
        )
    else:
        device_id = None

    # --------------------------------------------------------
    # Merchant
    # --------------------------------------------------------

    if transaction_type in [
        "MERCHANT_PAYMENT",
        "CARD_PAYMENT",
    ]:

        merchant_id = random.choice(
            merchant_map[country]
        )

    else:
        merchant_id = None

    # --------------------------------------------------------
    # Amount
    # --------------------------------------------------------

    typical_amount = profile[
        "typical_transaction_amount"
    ]

    amount = generate_amount(
        transaction_type,
        typical_amount
    )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    status = random.choices(
        STATUSES,
        weights=STATUS_WEIGHTS,
        k=1
    )[0]

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    transaction_time = generate_transaction_time(
        transaction_date
    )

    return {
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "merchant_id": merchant_id,
        "device_id": device_id,
        "transaction_time": transaction_time,
        "transaction_type": transaction_type,
        "channel": channel,
        "amount": amount,
        "currency": currency,
        "country": country,
        "city": city,
        "status": status,
    }


# ============================================================
# GENERATE TRANSACTIONS
# ============================================================

print("Generating transactions...")

transactions = []

transaction_counter = 1

current_date = START_DATE


while current_date <= END_DATE:

    # --------------------------------------------------------
    # Generate transactions for each customer
    # --------------------------------------------------------

    customer_ids = customers[
        "customer_id"
    ].tolist()

    for customer_id in customer_ids:

        profile = customer_profile_map[
            customer_id
        ]

        expected_daily_transactions = profile[
            "expected_daily_transactions"
        ]

        # ----------------------------------------------------
        # Sample number of transactions for this customer
        # on this particular day using Poisson distribution.
        # ----------------------------------------------------

        number_of_transactions = np.random.poisson(
            lam=expected_daily_transactions
        )

        # ----------------------------------------------------
        # Generate individual transactions
        # ----------------------------------------------------

        for _ in range(
            number_of_transactions
        ):

            transaction = generate_transaction(
                transaction_id=(
                    f"TXN_{transaction_counter:08d}"
                ),
                customer_id=customer_id,
                transaction_date=current_date,
            )

            transactions.append(
                transaction
            )

            transaction_counter += 1

    current_date += timedelta(
        days=1
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

transactions_df = pd.DataFrame(
    transactions
)


# ============================================================
# SAVE DATA
# ============================================================

output_path = (
    "data/synthetic/transactions.csv"
)

transactions_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\nTransaction generation complete.")

print(
    f"Number of transactions: "
    f"{len(transactions_df):,}"
)

print("\nTransaction type distribution:")

print(
    transactions_df[
        "transaction_type"
    ].value_counts()
)


print("\nChannel distribution:")

print(
    transactions_df[
        "channel"
    ].value_counts()
)


print("\nCurrency distribution:")

print(
    transactions_df[
        "currency"
    ].value_counts()
)


print("\nTransactions by activity profile:")

profile_summary = (
    transactions_df
    .merge(
        customer_profiles[
            [
                "customer_id",
                "activity_profile",
                "expected_daily_transactions",
            ]
        ],
        on="customer_id",
        how="left",
    )
    .groupby("activity_profile")
    .size()
)

print(
    profile_summary
)


print("\nAmount statistics by currency:")

print(
    transactions_df
    .groupby("currency")["amount"]
    .describe()
)


print(
    f"\nSaved to: {output_path}"
)