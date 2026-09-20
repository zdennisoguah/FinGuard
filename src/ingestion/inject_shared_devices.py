import random
import pandas as pd

SEED = 42
random.seed(SEED)

NUM_CUSTOMERS_IN_SCENARIO = 100

customers = pd.read_csv(
    "data/synthetic/customers.csv"
)

devices = pd.read_csv(
    "data/synthetic/devices.csv"
)

customer_devices = pd.read_csv(
    "data/synthetic/customer_devices.csv"
)
selected_customers = random.sample(
    customers["customer_id"].tolist(),
    NUM_CUSTOMERS_IN_SCENARIO,
)
random.shuffle(selected_customers)
clusters = []

remaining = selected_customers.copy()

while len(remaining) >= 2:

    cluster_size = min(
        random.randint(2, 6),
        len(remaining),
    )

    cluster = remaining[:cluster_size]

    remaining = remaining[cluster_size:]

    if len(cluster) >= 2:
        clusters.append(cluster)
shared_relationships = []

available_devices = devices[
    "device_id"
].tolist()

random.shuffle(available_devices)

for i, cluster in enumerate(clusters):

    device_id = available_devices[i]

    for customer_id in cluster:

        existing_link = (
            (customer_devices["customer_id"] == customer_id)
            &
            (customer_devices["device_id"] == device_id)
        )

        if not existing_link.any():

            shared_relationships.append({
                "customer_id": customer_id,
                "device_id": device_id,
                "first_seen": "2026-01-01",
            })
shared_relationships_df = pd.DataFrame(
    shared_relationships
)
customer_devices = pd.concat(
    [
        customer_devices,
        shared_relationships_df,
    ],
    ignore_index=True,
)
customer_devices.to_csv(
    "data/synthetic/customer_devices.csv",
    index=False,
)
