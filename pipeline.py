# pipeline.py
import warnings
import math
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import pulp

# Suppress warnings
warnings.filterwarnings("ignore")

DATA_PATH = "kerala_disease_climate_supply_chain.csv"

ITEM_PRIORITY = {
    "Oxygen_Cylinders": 1, "IV_Fluids": 2, "Mosquito_Nets": 3, "Paracetamol": 4, "Masks": 5,
}

MONSOON_MONTHS = {6, 7, 8, 9}
SERVICE_LEVEL_Z = 1.645

# Strategic Hub Definitions
WAREHOUSES = {
    "North_Hub_Kozhikode": {"lat": 11.2588, "lon": 75.7804, "capacity": 15000},
    "Central_Hub_Kochi": {"lat": 9.9312, "lon": 76.2673, "capacity": 20000},
    "South_Hub_Tvm": {"lat": 8.5241, "lon": 76.9366, "capacity": 15000},
}

DISTRICT_COORDS = {
    "Ernakulam": (9.9816, 76.2999), "Kozhikode": (11.2588, 75.7804),
    "Trivandrum": (8.5241, 76.9366), "Thrissur": (10.5276, 76.2144),
    "Kottayam": (9.5916, 76.5222), "Palakkad": (10.7867, 76.6548),
    "Alappuzha": (9.4981, 76.3388), "Wayanad": (11.6854, 76.1320),
    "Kollam": (8.8932, 76.6141), "Kannur": (11.8745, 75.3704),
    "Idukki": (9.8494, 76.9809), "Malappuram": (11.0510, 76.0711),
    "Kasaragod": (12.5102, 74.9852), "Pathanamthitta": (9.2648, 76.7870)
}


def _norm_item(x: str) -> str:
    if not isinstance(x, str): return x
    return x.strip().replace(" ", "_")


def load_data(path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=["Date", "District", "Item_Needed", "Patients_Visited"])
    df.columns = [c.strip().replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
    num_cols = ["TemperatureC", "Rainfallmm", "Humidity", "Patients_Visited", "Stock_Remaining", "Delay_Days"]
    for c in num_cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Item_Needed" in df.columns: df["Item_Needed"] = df["Item_Needed"].apply(_norm_item)
    df = df.sort_values(["District", "Item_Needed", "Date"]).reset_index(drop=True)
    df["Month"] = df["Date"].dt.month
    df["is_monsoon"] = df["Month"].isin(MONSOON_MONTHS).astype(int)
    return df.dropna(subset=["Date", "District", "Item_Needed", "Patients_Visited"])


# --- FEATURE: DISASTER SIMULATION ---
def inject_scenario_data(df: pd.DataFrame, scenario_type: str, severity: str, epicenter: str) -> pd.DataFrame:
    if scenario_type == "Normal Operations" or df.empty: return df
    df_sim = df.copy()
    mult_map = {"Low": 1.15, "Medium": 1.35, "High": 1.60, "Critical": 2.50}
    factor = mult_map.get(severity, 1.1)

    if scenario_type == "Viral Outbreak (Nipah/Dengue)":
        target_items = ["IV_Fluids", "Paracetamol", "Oxygen_Cylinders", "Masks"]
        is_target_item = df_sim["Item_Needed"].isin(target_items)
        is_epicenter = df_sim["District"] == epicenter
        df_sim.loc[is_target_item & is_epicenter, "Patients_Visited"] *= factor
        df_sim.loc[is_target_item & (~is_epicenter), "Patients_Visited"] *= (1 + (factor - 1) * 0.3)

    elif scenario_type == "Landslide/Flood (Infrastructure)":
        target_items = ["Mosquito_Nets", "Paracetamol", "IV_Fluids"]
        is_target_item = df_sim["Item_Needed"].isin(target_items)
        is_epicenter = df_sim["District"] == epicenter
        df_sim.loc[is_target_item & is_epicenter, "Patients_Visited"] *= factor

    elif scenario_type == "Festival/Mass Gathering":
        target_items = ["IV_Fluids", "Paracetamol"]
        is_target_item = df_sim["Item_Needed"].isin(target_items)
        is_epicenter = df_sim["District"] == epicenter
        df_sim.loc[is_target_item & is_epicenter, "Patients_Visited"] *= (factor * 0.8)

    return df_sim


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = []
    for (d, item), g in df.groupby(["District", "Item_Needed"], sort=False):
        g = g.sort_values("Date").copy()
        g["pv_lag1"] = g["Patients_Visited"].shift(1)
        g["pv_lag7"] = g["Patients_Visited"].shift(7)
        g["pv_7_mean"] = g["Patients_Visited"].rolling(7, min_periods=1).mean()
        g["dow"] = g["Date"].dt.dayofweek
        feats.append(g)
    return pd.concat(feats, ignore_index=True) if feats else pd.DataFrame()


def train_xgb(X_train, y_train):
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, n_jobs=0)
    model.fit(X_train, y_train)
    return model


def forecast_all(df_feat: pd.DataFrame, horizon=7):
    results = []
    for (d, item), g in df_feat.groupby(["District", "Item_Needed"]):
        # FIX: Lowered threshold from 15 to 8 so forecasts generate for more data
        if len(g) < 8: continue

        g = g.sort_values("Date")
        train = g.iloc[:-horizon]
        X = train[["pv_lag1", "pv_lag7", "pv_7_mean", "dow"]].fillna(0)
        y = train["Patients_Visited"]

        if X.empty: continue
        model = train_xgb(X, y)

        last_row = g.iloc[-1:].copy()
        future_X = pd.concat([last_row] * horizon, ignore_index=True)
        future_X = future_X[["pv_lag1", "pv_lag7", "pv_7_mean", "dow"]].fillna(0)
        yhat = model.predict(future_X)
        yhat = np.maximum(yhat, 0)

        sigma = np.std(y) if len(y) > 0 else 1.0
        results.append(pd.DataFrame(
            {"District": d, "Item_Needed": item, "t": np.arange(1, horizon + 1), "forecast": yhat, "sigma": sigma}))

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def detect_anomalies(df_feat: pd.DataFrame, k=2.0):
    alerts = []
    for (d, item), g in df_feat.groupby(["District", "Item_Needed"]):
        recent = g.sort_values("Date").tail(14)
        mu = recent["Patients_Visited"].mean()
        sigma = recent["Patients_Visited"].std() or 1.0
        threshold = mu + (k * sigma)
        for _, row in recent.iterrows():
            if row["Patients_Visited"] > threshold:
                alerts.append(
                    {"District": d, "Item_Needed": item, "Date": row["Date"], "Actual": row["Patients_Visited"],
                     "Threshold": threshold})
    return pd.DataFrame(alerts)


def compute_inventory_policies(df, fcst):
    stats = df.groupby(["District", "Item_Needed"])["Patients_Visited"].agg(["mean", "std"]).reset_index()
    policies = []
    for _, row in stats.iterrows():
        s = (row["mean"] * 2) + (SERVICE_LEVEL_Z * row["std"] * 1.414)
        S = s + (row["mean"] * 7)
        policies.append(
            {"District": row["District"], "Item_Needed": row["Item_Needed"], "reorder_point_s": s, "order_up_to_S": S})
    pol_df = pd.DataFrame(policies)
    if not fcst.empty:
        demand = fcst.groupby(["District", "Item_Needed"])["forecast"].sum().reset_index().rename(
            columns={"forecast": "expected_7d_demand"})
        pol_df = pol_df.merge(demand, on=["District", "Item_Needed"], how="left").fillna(0)
    return pol_df


def assign_warehouse(district):
    if district in ["Kasaragod", "Kannur", "Wayanad", "Kozhikode", "Malappuram"]:
        return "North_Hub_Kozhikode"
    elif district in ["Thrissur", "Palakkad", "Ernakulam", "Idukki"]:
        return "Central_Hub_Kochi"
    else:
        return "South_Hub_Tvm"


def optimize_allocation(fcst_sum: pd.DataFrame) -> pd.DataFrame:
    if fcst_sum.empty: return pd.DataFrame()
    allocs = []
    for _, row in fcst_sum.iterrows():
        allocs.append({"District": row["District"], "Item_Needed": row["Item_Needed"],
                       "alloc_qty": math.ceil(row["expected_7d_demand"]),
                       "Assigned_Hub": assign_warehouse(row["District"])})
    return pd.DataFrame(allocs)


def initialize_hub_stock():
    stock = {}
    items = ["Oxygen_Cylinders", "IV_Fluids", "Mosquito_Nets", "Paracetamol", "Masks"]
    for hub, info in WAREHOUSES.items():
        stock[hub] = {item: info["capacity"] // len(items) for item in items}
    return stock


def make_routes_and_update_stock(alloc: pd.DataFrame, hub_stocks: dict, scenario_type: str, epicenter: str):
    routes = []
    route_counter = 1
    is_logistics_disaster = "Landslide" in scenario_type or "Flood" in scenario_type
    for _, row in alloc.iterrows():
        qty = row["alloc_qty"]
        if qty <= 0: continue
        hub = row["Assigned_Hub"]
        item = row["Item_Needed"]
        dist = row["District"]
        status = "Dispatched"
        if hub in hub_stocks and item in hub_stocks[hub]:
            if hub_stocks[hub][item] >= qty:
                hub_stocks[hub][item] -= qty
            else:
                if hub != "Central_Hub_Kochi" and hub_stocks["Central_Hub_Kochi"].get(item, 0) >= qty:
                    hub = "Central_Hub_Kochi"
                    hub_stocks["Central_Hub_Kochi"][item] -= qty
                    status = "Rerouted (Regional Stockout)"
                else:
                    status = "Backordered (Network Stockout)"
        if is_logistics_disaster and dist == epicenter: status = "Route Blocked / Delayed"
        routes.append({"Route_ID": f"RT-{route_counter:03d}", "Origin_Hub": hub, "Destination": dist, "Item": item,
                       "Quantity": qty, "Status": status})
        route_counter += 1
    return pd.DataFrame(routes), hub_stocks