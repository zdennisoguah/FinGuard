import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_TRANSACTIONS_PATH = "data/synthetic/transactions.csv"

VELOCITY_FRAUD_PATH = "data/synthetic/fraud_events.csv"
ATO_FRAUD_PATH = "data/synthetic/ato_transactions.csv"
RING_FRAUD_PATH = "data/synthetic/fraud_ring_transactions.csv"

GROUND_TRUTH_PATH = "data/synthetic/fraud_ground_truth.csv"
OBSERVED_TRANSACTIONS_PATH = "data/synthetic/transactions_observed.csv"


OPERATIONAL_COLUMNS = [
    "transaction_id",
    "customer_id",
    "merchant_id",
    "device_id",
    "transaction_time",
    "transaction_type",
    "channel",
    "amount",
    "currency",
    "country",
    "city",
    "status",
]


GROUND_TRUTH_COLUMNS = [
    "transaction_id",
    "customer_id",
    "scenario_id",
    "scenario",
    "fraud_label",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("Loading baseline transactions...")

    baseline = pd.read_csv(
        BASE_TRANSACTIONS_PATH,
        parse_dates=["transaction_time"]
    )

    print(f"Baseline transactions: {len(baseline):,}")

    print("\nLoading velocity fraud...")

    velocity = pd.read_csv(
        VELOCITY_FRAUD_PATH,
        parse_dates=["transaction_time"]
    )

    print(f"Velocity fraud transactions: {len(velocity):,}")

    print("\nLoading account takeover fraud...")

    ato = pd.read_csv(
        ATO_FRAUD_PATH,
        parse_dates=["transaction_time"]
    )

    print(f"Account takeover transactions: {len(ato):,}")

    print("\nLoading fraud-ring transactions...")

    ring = pd.read_csv(
        RING_FRAUD_PATH,
        parse_dates=["transaction_time"]
    )

    print(f"Fraud-ring transactions: {len(ring):,}")

    return baseline, velocity, ato, ring


# ============================================================
# VALIDATE FRAUD SOURCES
# ============================================================

def validate_fraud_sources(
    baseline,
    velocity,
    ato,
    ring
):

    print("\nValidating fraud sources...")

    fraud_sources = {
        "VELOCITY": velocity,
        "ACCOUNT_TAKEOVER": ato,
        "FRAUD_RING": ring,
    }

    # --------------------------------------------------------
    # Check fraud labels
    # --------------------------------------------------------

    for name, df in fraud_sources.items():

        assert "fraud_label" in df.columns, (
            f"{name} is missing fraud_label"
        )

        assert (
            df["fraud_label"] == 1
        ).all(), (
            f"{name} contains non-fraud labels"
        )

    print("Fraud labels: PASS")

    # --------------------------------------------------------
    # Display actual scenarios
    # --------------------------------------------------------

    print("\nScenarios found in source files:")

    for name, df in fraud_sources.items():

        print(f"{name}:")
        print(df["scenario"].value_counts())

    # --------------------------------------------------------
    # Normalize scenario names
    # --------------------------------------------------------

    scenario_mapping = {
        "VELOCITY": "VELOCITY",
        "VELOCITY_FRAUD": "VELOCITY",

        "ACCOUNT_TAKEOVER": "ACCOUNT_TAKEOVER",
        "ACCOUNT_TAKEOVER_FRAUD": "ACCOUNT_TAKEOVER",

        "FRAUD_RING": "FRAUD_RING",
        "FRAUD_RING_FRAUD": "FRAUD_RING",
    }

    for name, df in fraud_sources.items():

        df["scenario"] = (
            df["scenario"]
            .map(scenario_mapping)
            .fillna(df["scenario"])
        )

    # --------------------------------------------------------
    # Validate canonical scenarios
    # --------------------------------------------------------

    expected_scenarios = {
        "VELOCITY": "VELOCITY",
        "ACCOUNT_TAKEOVER": "ACCOUNT_TAKEOVER",
        "FRAUD_RING": "FRAUD_RING",
    }

    for name, expected in expected_scenarios.items():

        df = fraud_sources[name]

        assert (
            df["scenario"] == expected
        ).all(), (
            f"{name} contains unexpected scenarios "
            f"after normalization"
        )

    print("Fraud scenarios: PASS")

    # --------------------------------------------------------
    # Check fraud IDs are unique within each source
    # --------------------------------------------------------

    for name, df in fraud_sources.items():

        assert df["transaction_id"].is_unique, (
            f"{name} contains duplicate transaction IDs"
        )

    print("Within-source transaction IDs: PASS")

    # --------------------------------------------------------
    # Check fraud IDs do not overlap baseline
    # --------------------------------------------------------

    baseline_ids = set(
        baseline["transaction_id"]
    )

    for name, df in fraud_sources.items():

        overlap = baseline_ids.intersection(
            set(df["transaction_id"])
        )

        assert len(overlap) == 0, (
            f"{name} overlaps with baseline transactions"
        )

    print("Baseline/fraud ID separation: PASS")

    # --------------------------------------------------------
    # Check fraud sources do not overlap each other
    # --------------------------------------------------------

    fraud_names = list(fraud_sources.keys())

    for i in range(len(fraud_names)):

        for j in range(i + 1, len(fraud_names)):

            name_a = fraud_names[i]
            name_b = fraud_names[j]

            ids_a = set(
                fraud_sources[name_a]["transaction_id"]
            )

            ids_b = set(
                fraud_sources[name_b]["transaction_id"]
            )

            overlap = ids_a.intersection(ids_b)

            assert len(overlap) == 0, (
                f"{name_a} and {name_b} have "
                f"duplicate transaction IDs"
            )

    print("Fraud-source ID separation: PASS")


# ============================================================
# BUILD GROUND TRUTH
# ============================================================

def build_ground_truth(
    velocity,
    ato,
    ring
):

    print("\nBuilding consolidated fraud ground truth...")

    velocity_gt = velocity[
        GROUND_TRUTH_COLUMNS
    ].copy()

    ato_gt = ato[
        GROUND_TRUTH_COLUMNS
    ].copy()

    ring_gt = ring[
        GROUND_TRUTH_COLUMNS
    ].copy()

    ground_truth = pd.concat(
        [
            velocity_gt,
            ato_gt,
            ring_gt,
        ],
        ignore_index=True
    )

    ground_truth = (
        ground_truth
        .sort_values("transaction_id")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    assert ground_truth["transaction_id"].is_unique

    assert (
        ground_truth["fraud_label"] == 1
    ).all()

    print(
        f"Total fraud transactions: "
        f"{len(ground_truth):,}"
    )

    print("\nFraud by scenario:")

    print(
        ground_truth["scenario"]
        .value_counts()
    )

    ground_truth.to_csv(
        GROUND_TRUTH_PATH,
        index=False
    )

    print(
        f"\nSaved ground truth to: "
        f"{GROUND_TRUTH_PATH}"
    )

    return ground_truth


# ============================================================
# BUILD OBSERVED DATASET
# ============================================================

def build_observed_dataset(
    baseline,
    velocity,
    ato,
    ring
):

    print("\nBuilding observed transaction dataset...")

    # --------------------------------------------------------
    # Keep only operational columns
    # --------------------------------------------------------

    velocity_observed = velocity[
        OPERATIONAL_COLUMNS
    ].copy()

    ato_observed = ato[
        OPERATIONAL_COLUMNS
    ].copy()

    ring_observed = ring[
        OPERATIONAL_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # Combine baseline + fraud transactions
    # --------------------------------------------------------

    observed = pd.concat(
        [
            baseline[OPERATIONAL_COLUMNS],
            velocity_observed,
            ato_observed,
            ring_observed,
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    observed = (
        observed
        .sort_values("transaction_time")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Validate operational dataset
    # --------------------------------------------------------

    assert observed["transaction_id"].is_unique

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

    assert list(observed.columns) == OPERATIONAL_COLUMNS

    print(
        f"Observed transactions: "
        f"{len(observed):,}"
    )

    print("Operational columns only: PASS")

    observed.to_csv(
        OBSERVED_TRANSACTIONS_PATH,
        index=False
    )

    print(
        f"Saved observed dataset to: "
        f"{OBSERVED_TRANSACTIONS_PATH}"
    )

    return observed


# ============================================================
# FINAL VALIDATION
# ============================================================

def final_validation(
    baseline,
    ground_truth,
    observed
):

    print("\nRunning final validation...")

    expected_observed_count = (
        len(baseline)
        + len(ground_truth)
    )

    assert (
        len(observed) == expected_observed_count
    ), (
        "Observed transaction count does not "
        "match baseline + fraud transactions"
    )

    print(
        "Observed row count: PASS"
    )

    observed_ids = set(
        observed["transaction_id"]
    )

    ground_truth_ids = set(
        ground_truth["transaction_id"]
    )

    missing_fraud_ids = (
        ground_truth_ids - observed_ids
    )

    assert len(missing_fraud_ids) == 0, (
        "Some fraud transactions are missing "
        "from observed data"
    )

    print(
        "All fraud transactions present in observed data: PASS"
    )

    assert (
        observed["transaction_id"].is_unique
    )

    print(
        "Observed transaction IDs unique: PASS"
    )

    print("\nFinal dataset statistics:")

    print(
        f"Baseline transactions: "
        f"{len(baseline):,}"
    )

    print(
        f"Fraud transactions: "
        f"{len(ground_truth):,}"
    )

    print(
        f"Observed transactions: "
        f"{len(observed):,}"
    )

    print("\nFraud distribution:")

    print(
        ground_truth["scenario"]
        .value_counts()
    )

    print("\nAll final validations passed.")


# ============================================================
# MAIN
# ============================================================

def main():

    baseline, velocity, ato, ring = load_data()

    validate_fraud_sources(
        baseline,
        velocity,
        ato,
        ring
    )

    ground_truth = build_ground_truth(
        velocity,
        ato,
        ring
    )

    observed = build_observed_dataset(
        baseline,
        velocity,
        ato,
        ring
    )

    final_validation(
        baseline,
        ground_truth,
        observed
    )


if __name__ == "__main__":
    main()