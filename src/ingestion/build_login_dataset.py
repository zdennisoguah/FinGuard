import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

NORMAL_LOGIN_PATH = "data/synthetic/login_events.csv"
ATO_LOGIN_PATH = "data/synthetic/ato_events.csv"

OBSERVED_LOGIN_PATH = (
    "data/synthetic/login_events_observed.csv"
)

GROUND_TRUTH_LOGIN_PATH = (
    "data/synthetic/login_ground_truth.csv"
)


OBSERVED_COLUMNS = [
    "login_id",
    "customer_id",
    "device_id",
    "login_time",
    "ip_address",
    "country",
    "city",
    "login_status",
    "authentication_method",
    "event_type",
]


GROUND_TRUTH_COLUMNS = [
    "login_id",
    "customer_id",
    "scenario_id",
    "event_type",
    "fraud_label",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading normal login events...")

    normal_logins = pd.read_csv(
        NORMAL_LOGIN_PATH,
        parse_dates=["login_time"]
    )

    print(
        f"Normal login events: "
        f"{len(normal_logins):,}"
    )

    print("\nLoading account takeover login events...")

    ato_logins = pd.read_csv(
        ATO_LOGIN_PATH,
        parse_dates=["event_time"]
    )

    print(
        f"ATO login events: "
        f"{len(ato_logins):,}"
    )

    print("\nNormal login columns:")
    print(normal_logins.columns.tolist())

    print("\nATO login columns:")
    print(ato_logins.columns.tolist())

    return normal_logins, ato_logins


# ============================================================
# NORMALIZE NORMAL LOGIN EVENTS
# ============================================================

def normalize_normal_logins(normal_logins):

    normal = normal_logins.copy()

    normal["event_type"] = "LOGIN"

    normal = normal.rename(
        columns={
            "login_status": "login_status",
            "authentication_method": "authentication_method",
        }
    )

    normal = normal[
        [
            "login_id",
            "customer_id",
            "device_id",
            "login_time",
            "ip_address",
            "country",
            "city",
            "login_status",
            "authentication_method",
            "event_type",
        ]
    ]

    return normal


# ============================================================
# NORMALIZE ATO LOGIN EVENTS
# ============================================================

def normalize_ato_logins(ato_logins):

    ato = ato_logins.copy()

    # --------------------------------------------------------
    # Create operational login ID
    # --------------------------------------------------------

    ato["login_id"] = ato["event_id"]

    # --------------------------------------------------------
    # Rename event timestamp
    # --------------------------------------------------------

    ato["login_time"] = ato["event_time"]

    # --------------------------------------------------------
    # ATO events do not currently contain country/city.
    # These will be null rather than fabricated.
    # --------------------------------------------------------

    ato["country"] = pd.NA
    ato["city"] = pd.NA

    # --------------------------------------------------------
    # Select operational columns
    # --------------------------------------------------------

    ato = ato[
        [
            "login_id",
            "customer_id",
            "device_id",
            "login_time",
            "ip_address",
            "country",
            "city",
            "login_status",
            "authentication_method",
            "event_type",
        ]
    ]

    return ato


# ============================================================
# VALIDATE SOURCE DATA
# ============================================================

def validate_sources(
    normal_logins,
    ato_logins
):

    print("\nValidating login sources...")

    # --------------------------------------------------------
    # Source IDs
    # --------------------------------------------------------

    assert normal_logins["login_id"].is_unique
    assert ato_logins["event_id"].is_unique

    print("Source event IDs unique: PASS")

    # --------------------------------------------------------
    # ATO ground truth columns
    # --------------------------------------------------------

    required_ato_columns = [
        "event_id",
        "scenario_id",
        "customer_id",
        "event_type",
        "event_time",
        "device_id",
        "ip_address",
        "login_status",
        "authentication_method",
    ]

    for column in required_ato_columns:

        assert column in ato_logins.columns, (
            f"ATO login data missing column: {column}"
        )

    print("ATO source schema: PASS")

    # --------------------------------------------------------
    # ATO scenario
    # --------------------------------------------------------

    assert (
        ato_logins["scenario_id"]
        .astype(str)
        .str.startswith("ATO_")
    ).all()

    print("ATO scenario IDs: PASS")

    # --------------------------------------------------------
    # No ID collision between normal and ATO events
    # --------------------------------------------------------

    normal_ids = set(
        normal_logins["login_id"]
    )

    ato_ids = set(
        ato_logins["event_id"]
    )

    overlap = normal_ids.intersection(
        ato_ids
    )

    print(
        f"Normal/ATO event ID overlap: {len(overlap)}"
    )

    assert len(overlap) == 0

    print("Event ID separation: PASS")


# ============================================================
# BUILD LOGIN GROUND TRUTH
# ============================================================

def build_ground_truth(ato_logins):

    print("\nBuilding login ground truth...")

    ground_truth = pd.DataFrame({
        "login_id": ato_logins["event_id"],
        "customer_id": ato_logins["customer_id"],
        "scenario_id": ato_logins["scenario_id"],
        "event_type": ato_logins["event_type"],
        "fraud_label": 1,
    })

    assert ground_truth["login_id"].is_unique

    assert (
        ground_truth["fraud_label"] == 1
    ).all()

    ground_truth.to_csv(
        GROUND_TRUTH_LOGIN_PATH,
        index=False
    )

    print(
        f"Login ground truth events: "
        f"{len(ground_truth):,}"
    )

    print("\nATO event types:")

    print(
        ground_truth["event_type"]
        .value_counts()
    )

    print(
        f"\nSaved to: "
        f"{GROUND_TRUTH_LOGIN_PATH}"
    )

    return ground_truth


# ============================================================
# BUILD OBSERVED LOGIN DATASET
# ============================================================

def build_observed_dataset(
    normal_logins,
    ato_logins
):

    print("\nNormalizing login events...")

    normal = normalize_normal_logins(
        normal_logins
    )

    ato = normalize_ato_logins(
        ato_logins
    )

    print(
        f"Normalized normal events: "
        f"{len(normal):,}"
    )

    print(
        f"Normalized ATO events: "
        f"{len(ato):,}"
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    observed = pd.concat(
        [
            normal,
            ato,
        ],
        ignore_index=True
    )

    observed = (
        observed
        .sort_values("login_time")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Validate schema
    # --------------------------------------------------------

    assert list(
        observed.columns
    ) == OBSERVED_COLUMNS

    # --------------------------------------------------------
    # IDs must be unique
    # --------------------------------------------------------

    assert observed["login_id"].is_unique

    # --------------------------------------------------------
    # No fraud metadata
    # --------------------------------------------------------

    forbidden_columns = {
        "fraud_label",
        "scenario",
        "scenario_id",
    }

    leaked_columns = (
        forbidden_columns
        .intersection(observed.columns)
    )

    assert len(leaked_columns) == 0, (
        f"Fraud metadata leaked into observed data: "
        f"{leaked_columns}"
    )

    print("Operational schema: PASS")

    print("Fraud metadata leakage check: PASS")

    observed.to_csv(
        OBSERVED_LOGIN_PATH,
        index=False
    )

    print(
        f"\nObserved login events: "
        f"{len(observed):,}"
    )

    print(
        f"Saved to: "
        f"{OBSERVED_LOGIN_PATH}"
    )

    return observed


# ============================================================
# FINAL VALIDATION
# ============================================================

def final_validation(
    normal_logins,
    ground_truth,
    observed
):

    print("\nRunning final validation...")

    # --------------------------------------------------------
    # Expected row count
    # --------------------------------------------------------

    expected_count = (
        len(normal_logins)
        + len(ground_truth)
    )

    assert (
        len(observed) == expected_count
    )

    print("Observed row count: PASS")

    # --------------------------------------------------------
    # All fraud login IDs must exist
    # --------------------------------------------------------

    observed_ids = set(
        observed["login_id"]
    )

    ground_truth_ids = set(
        ground_truth["login_id"]
    )

    missing_ids = (
        ground_truth_ids - observed_ids
    )

    assert len(missing_ids) == 0

    print(
        "All ATO login events present "
        "in observed data: PASS"
    )

    # --------------------------------------------------------
    # Unique IDs
    # --------------------------------------------------------

    assert observed["login_id"].is_unique

    print(
        "Observed login IDs unique: PASS"
    )

    # --------------------------------------------------------
    # ATO event distribution
    # --------------------------------------------------------

    ato_events = observed[
        observed["login_id"].isin(
            ground_truth_ids
        )
    ]

    print("\nATO events in observed data:")

    print(
        ato_events["event_type"]
        .value_counts()
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\nFinal login dataset:")

    print(
        f"Normal login events: "
        f"{len(normal_logins):,}"
    )

    print(
        f"ATO login events: "
        f"{len(ground_truth):,}"
    )

    print(
        f"Observed login events: "
        f"{len(observed):,}"
    )

    print("\nAll login dataset validations passed.")


# ============================================================
# MAIN
# ============================================================

def main():

    normal_logins, ato_logins = load_data()

    validate_sources(
        normal_logins,
        ato_logins
    )

    ground_truth = build_ground_truth(
        ato_logins
    )

    observed = build_observed_dataset(
        normal_logins,
        ato_logins
    )

    final_validation(
        normal_logins,
        ground_truth,
        observed
    )


if __name__ == "__main__":
    main()