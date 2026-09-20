import random
from datetime import timedelta

import pandas as pd


SEED = 42

random.seed(SEED)


SIMULATION_END_DATE = pd.Timestamp("2026-08-31")


DEVICE_TYPES = {
    "Mobile": 0.75,
    "Desktop": 0.20,
    "Tablet": 0.05,
}


OPERATING_SYSTEMS = {
    "Mobile": ["Android", "iOS"],
    "Desktop": ["Windows", "macOS", "Linux"],
    "Tablet": ["Android", "iPadOS"],
}


BROWSERS = [
    "Chrome",
    "Safari",
    "Firefox",
    "Edge",
]
def number_of_devices():

    return random.choices(
        [1, 2, 3, 4],
        weights=[0.70, 0.20, 0.08, 0.02],
        k=1
    )[0]
def generate_device_id(index):

    return f"DEV_{index:06d}"
def generate_devices(customers):

    devices = []
    customer_device_links = []

    device_counter = 1

    for _, customer in customers.iterrows():

        customer_id = customer["customer_id"]

        account_created_at = pd.to_datetime(
            customer["account_created_at"]
        )

        num_devices = number_of_devices()

        for _ in range(num_devices):

            device_id = f"DEV_{device_counter:06d}"

            device_type = random.choices(
                list(DEVICE_TYPES.keys()),
                weights=list(DEVICE_TYPES.values()),
                k=1,
            )[0]

            operating_system = random.choice(
                OPERATING_SYSTEMS[device_type]
            )

            browser = random.choice(BROWSERS)

            days_after_signup = random.randint(
                0,
                max(
                    1,
                    (
                        SIMULATION_END_DATE
                        - account_created_at
                    ).days,
                ),
            )

            first_seen = (
                account_created_at
                + timedelta(days=days_after_signup)
            )

            devices.append({
                "device_id": device_id,
                "device_type": device_type,
                "operating_system": operating_system,
                "browser": browser,
            })

            customer_device_links.append({
                "customer_id": customer_id,
                "device_id": device_id,
                "first_seen": first_seen,
            })

            device_counter += 1

    return (
        pd.DataFrame(devices),
        pd.DataFrame(customer_device_links),
    )
if __name__ == "__main__":

    customers = pd.read_csv(
        "data/synthetic/customers.csv"
    )

    devices, customer_devices = generate_devices(
        customers
    )

    print("Devices:")
    print(devices.head())

    print()

    print("Customer-device relationships:")
    print(customer_devices.head())

    print()

    print(
        f"Number of devices: {len(devices):,}"
    )

    print(
        f"Customer-device relationships: "
        f"{len(customer_devices):,}"
    )

    devices.to_csv(
        "data/synthetic/devices.csv",
        index=False,
    )

    customer_devices.to_csv(
        "data/synthetic/customer_devices.csv",
        index=False,
    )