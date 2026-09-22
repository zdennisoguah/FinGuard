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

MODE = "DEV"

DEV_START_DATE = datetime(2026, 1, 1)
DEV_END_DATE = datetime(2026, 1, 30)

FULL_START_DATE = datetime(2026, 1, 1)
FULL_END_DATE = datetime(2026, 8, 31)


# ============================================================
# ACTIVE DATE RANGE
# ============================================================

if MODE == "DEV":
    START_DATE = DEV_START_DATE
    END_DATE = DEV_END_DATE

elif MODE == "FULL":
    START_DATE = FULL_START_DATE
    END_DATE = FULL_END_DATE

else:
    raise ValueError(
        "MODE must be either 'DEV' or 'FULL'."
    )


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
# TIME-OF-DAY DISTRIBUTIONS
# ============================================================

TIME_BUCKETS = {
    "NIGHT": (0, 5),
    "MORNING": (6, 8),
    "DAYTIME": (9, 16),
    "EVENING": (17, 20),
    "LATE_EVENING": (21, 23),
}


TIME_WEIGHTS_BY_TYPE = {
    "P2P_TRANSFER": {
        "NIGHT": 0.05,
        "MORNING": 0.15,
        "DAYTIME": 0.40,
        "EVENING": 0.30,
        "LATE_EVENING": 0.10,
    },

    "MERCHANT_PAYMENT": {
        "NIGHT": 0.02,
        "MORNING": 0.15,
        "DAYTIME": 0.45,
        "EVENING": 0.33,
        "LATE_EVENING": 0.05,
    },

    "CARD_PAYMENT": {
        "NIGHT": 0.03,
        "MORNING": 0.15,
        "DAYTIME": 0.45,
        "EVENING": 0.32,
        "LATE_EVENING": 0.05,
    },

    "CASH_WITHDRAWAL": {
        "NIGHT": 0.01,
        "MORNING": 0.20,
        "DAYTIME": 0.50,
        "EVENING": 0.27,
        "LATE_EVENING": 0.02,
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

customer_profiles[
    "expected_daily_transactions"
] = (
    customer_profiles[
        "expected_daily_transactions"
    ].astype(float)
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


customer_ids = customers[
    "customer_id"
].tolist()


# ============================================================
# SELECT TIME BUCKET
# ============================================================

def select_time_bucket(transaction_type):

    buckets = list(
        TIME_WEIGHTS_BY_TYPE[
            transaction_type
        ].keys()
    )

    probabilities = list(
        TIME_WEIGHTS_BY_TYPE[
            transaction_type
        ].values()
    )

    return random.choices(
        buckets,
        weights=probabilities,
        k=1
    )[0]


# ============================================================
# GENERATE TRANSACTION TIME
# ============================================================

def generate_transaction_time(
    transaction_date,
    transaction_type
):
    """
    Generate a transaction timestamp using
    transaction-type-specific time-of-day behavior.
    """

    time_bucket = select_time_bucket(
        transaction_type
    )

    start_hour, end_hour = (
        TIME_BUCKETS[time_bucket]
    )

    hour = random.randint(
        start_hour,
        end_hour
    )

    minute = random.randint(
        0,
        59
    )

    second = random.randint(
        0,
        59
    )

    return transaction_date + timedelta(
        hours=hour,
        minutes=minute,
        seconds=second,
    )


# ============================================================
# GENERATE TRANSACTION AMOUNT
# ============================================================

def generate_amount(
    transaction_type,
    typical_amount
):

    if transaction_type == "CASH_WITHDRAWAL":
        sigma = 0.45

    elif transaction_type == "CARD_PAYMENT":
        sigma = 0.35

    elif transaction_type == "MERCHANT_PAYMENT":
        sigma = 0.40

    else:
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

    customer = customer_map[
        customer_id
    ]

    profile = customer_profile_map[
        customer_id
    ]

    country = customer[
        "country"
    ]

    city = customer[
        "city"
    ]

    currency = profile[
        "currency"
    ]

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

    transaction_time = (
        generate_transaction_time(
            transaction_date,
            transaction_type
        )
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

print(
    f"Generating transactions in {MODE} mode..."
)

transactions = []

transaction_counter = 1

current_date = START_DATE


while current_date <= END_DATE:

    for customer_id in customer_ids:

        profile = customer_profile_map[
            customer_id
        ]

        expected_daily_transactions = profile[
            "expected_daily_transactions"
        ]

        number_of_transactions = np.random.poisson(
            lam=expected_daily_transactions
        )

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
# SAVE
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
    f"Mode: {MODE}"
)

print(
    f"Date range: "
    f"{START_DATE.date()} → {END_DATE.date()}"
)

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

print("\nTransaction hour distribution:")

print(
    pd.to_datetime(
        transactions_df[
            "transaction_time"
        ]
    )
    .dt.hour
    .value_counts()
    .sort_index()
)

print(
    f"\nSaved to: {output_path}"
)