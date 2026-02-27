# app.py
import streamlit as st

# Must be the very first command
st.set_page_config(page_title="Smart Medical Supply Chain", layout="wide")

import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import math

# Import Backend
from pipeline import (
    DATA_PATH, load_data, inject_scenario_data, make_features, forecast_all,
    detect_anomalies, compute_inventory_policies, optimize_allocation,
    make_routes_and_update_stock, initialize_hub_stock,
    WAREHOUSES, DISTRICT_COORDS
)

# ==========================================
# INITIALIZE SESSION STATE
# ==========================================
if "warehouse_data" not in st.session_state:
    # Initialize stock matching the pipeline definitions
    initial_stock = {}
    items = ["Oxygen_Cylinders", "IV_Fluids", "Mosquito_Nets", "Paracetamol", "Masks"]
    for hub_key, info in WAREHOUSES.items():
        initial_stock[hub_key] = {
            "lat": info["lat"], "lon": info["lon"],
            "stock": {item: info["capacity"] // len(items) for item in items}
        }
    st.session_state.warehouse_data = initial_stock

if "manual_dispatches" not in st.session_state:
    st.session_state.manual_dispatches = []


# ---------- Cached pipeline run ----------
@st.cache_data(show_spinner=True)
def run_pipeline(data_path: Path, event_type: str, severity: str, epicenter: str):
    # 1. Load Data
    df_raw = load_data(data_path)

    # 2. Simulation Injection (New Feature)
    df_sim = inject_scenario_data(df_raw, event_type, severity, epicenter)

    # 3. Standard Pipeline
    df_feat = make_features(df_sim)
    fcst = forecast_all(df_feat, horizon=7)
    # Corrected call matching pipeline.py
    alerts = detect_anomalies(df_feat, k=1.0)
    policies = compute_inventory_policies(df_sim, fcst)

    fcst_sum = policies[["District", "Item_Needed", "expected_7d_demand"]]
    alloc = optimize_allocation(fcst_sum)

    # 4. Generate Auto Routes (calculation only)
    routes, _ = make_routes_and_update_stock(alloc, initialize_hub_stock(), event_type, epicenter)

    return df_sim, df_feat, fcst, alerts, policies, alloc, routes


def load_outputs_if_exist():
    out = Path("outputs")
    if not out.exists(): return {}

    def maybe(p):
        f = out / p
        return pd.read_csv(f) if f.exists() else None

    return dict(fcst=maybe("forecast_7d.csv"), alerts=maybe("anomalies.csv"),
                policies=maybe("inventory_policies.csv"), alloc=maybe("allocation.csv"),
                routes=maybe("routes.csv"))


# ---------- Sidebar ----------
st.sidebar.title("Controls")

# 1. Data Source
st.sidebar.header("1. Data Configuration")
mode = st.sidebar.radio("Data source", ["Run pipeline now", "Load last outputs"])
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
data_path_file = uploaded if uploaded is not None else DATA_PATH

st.sidebar.markdown("---")

# 2. Disaster Simulation (New Feature)
st.sidebar.header("2. Disaster Simulation")
st.sidebar.info("Select a scenario to test system response.")
sim_mode = st.sidebar.selectbox("Scenario Type",
                                ["Normal Operations", "Viral Outbreak (Nipah/Dengue)",
                                 "Landslide/Flood (Infrastructure)", "Festival/Mass Gathering"])
sim_epicenter = st.sidebar.selectbox("Epicenter District", list(DISTRICT_COORDS.keys()))
sim_severity = st.sidebar.select_slider("Event Severity", options=["Low", "Medium", "High", "Critical"])

run_btn = st.sidebar.button("Execute / Refresh")

if mode == "Run pipeline now" and run_btn:
    with st.spinner("Running Pipeline..."):
        df, df_feat, fcst, alerts, policies, alloc, routes = run_pipeline(data_path_file, sim_mode, sim_severity,
                                                                          sim_epicenter)
        st.session_state.update(dict(
            df=df, df_feat=df_feat, fcst=fcst, alerts=alerts,
            policies=policies, alloc=alloc, routes=routes
        ))
elif mode == "Load last outputs" and run_btn:
    st.session_state.update(load_outputs_if_exist())

st.title("Smart Medical Supply Chain Dashboard")

# Confirmation Message logic
if "dispatch_success" in st.session_state:
    st.success(st.session_state.dispatch_success)
    del st.session_state.dispatch_success


def apply_filters(df, district, item):
    if df is None or (hasattr(df, "empty") and df.empty): return df
    x = df.copy()
    if "District" in x.columns and district:
        x = x[x["District"].astype(str).str.contains(district, case=False, na=False)]
    if "Item_Needed" in x.columns and item:
        x = x[x["Item_Needed"].astype(str).str.contains(item, case=False, na=False)]
    return x


c1, c2 = st.columns(2)
district = c1.text_input("Filter: District", "")
item = c2.text_input("Filter: Item", "")

tabs = st.tabs(
    ["Forecasts", "Anomalies", "Inventory", "Allocation", "Routes", "Inventory Update / Dispatch Map", "About"])

# ---------- Forecasts tab (Original Layout) ----------
with tabs[0]:
    st.subheader("7-Day Forecasts with confidence band")
    fcst = apply_filters(st.session_state.get("fcst"), district, item)
    df = st.session_state.get("df")

    if fcst is None or df is None or fcst.empty:
        st.info("Click Execute / Refresh to generate forecasts.")
    else:
        pick = fcst.groupby(["District", "Item_Needed"]).size().reset_index().drop(columns=0)
        sel = st.selectbox("Select District-Item", options=[(r.District, r.Item_Needed) for _, r in pick.iterrows()])

        # Original Chart Logic
        dsel = fcst[(fcst["District"] == sel[0]) & (fcst["Item_Needed"] == sel[1])].copy()
        if "sigma" in dsel.columns:
            dsel["upper"] = dsel["forecast"] + 2.0 * dsel["sigma"]
            dsel["lower"] = (dsel["forecast"] - 2.0 * dsel["sigma"]).clip(lower=0.0)

        recent = df[(df["District"] == sel[0]) & (df["Item_Needed"] == sel[1])].sort_values("Date").tail(30)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=recent["Date"], y=recent["Patients_Visited"], mode="lines+markers", name="Actual (last 30d)"))

        start = recent["Date"].max() if not recent.empty else None
        future_dates = pd.date_range(start=start + pd.Timedelta(days=1), periods=len(dsel),
                                     freq="D") if start is not None else pd.RangeIndex(1, len(dsel) + 1)

        fig.add_trace(go.Scatter(x=future_dates, y=dsel["forecast"], mode="lines+markers", name="Forecast 7d"))
        if "upper" in dsel:
            fig.add_trace(go.Scatter(x=future_dates, y=dsel["upper"], line=dict(width=0), showlegend=False))
            fig.add_trace(
                go.Scatter(x=future_dates, y=dsel["lower"], fill="tonexty", line=dict(width=0), name="Conf +/- 2 sigma",
                           hoverinfo="skip"))

        fig.update_layout(height=450, xaxis_title="Date", yaxis_title="Demand")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(dsel, use_container_width=True)

# ---------- Anomalies tab (Original Layout) ----------
with tabs[1]:
    st.subheader("Anomalies")
    alerts = apply_filters(st.session_state.get("alerts"), district, item)
    if alerts is None or alerts.empty:
        st.info("No anomalies found.")
    else:
        st.dataframe(alerts.sort_values("Date", ascending=False), use_container_width=True, height=360)
        agg = alerts.groupby(["District", "Item_Needed"]).size().reset_index(name="count")
        if not agg.empty:
            st.plotly_chart(px.bar(agg.sort_values("count", ascending=False).head(20),
                                   x="count", y="District", color="Item_Needed", orientation="h",
                                   title="Top anomaly counts"),
                            use_container_width=True)

# ---------- Inventory tab ----------
with tabs[2]:
    st.subheader("Inventory policy (s, S)")
    pol = apply_filters(st.session_state.get("policies"), district, item)
    if pol is None or pol.empty:
        st.info("No policy data yet.")
    else:
        st.dataframe(pol, use_container_width=True, height=360)

# ---------- Allocation tab ----------
with tabs[3]:
    st.subheader("Allocation plan")
    alloc = apply_filters(st.session_state.get("alloc"), district, item)
    if alloc is None or alloc.empty:
        st.info("No allocation data yet.")
    else:
        st.dataframe(alloc.sort_values("alloc_qty", ascending=False), use_container_width=True, height=360)

# ---------- Routes tab ----------
with tabs[4]:
    st.subheader("Routes (greedy clusters)")
    routes = st.session_state.get("routes")
    if routes is None or (hasattr(routes, "empty") and routes.empty):
        st.info("No routes yet.")
    else:
        st.dataframe(routes, use_container_width=True, height=360)

# ---------- Inventory Update / Dispatch Map tab (NEW FEATURES HERE) ----------
with tabs[5]:
    st.subheader("Warehouse & Emergency Dispatch")

    warehouses = st.session_state.warehouse_data
    all_items = sorted({i for w in warehouses.values() for i in w["stock"].keys()})

    # 1. Manual Stock Update (Interactive)
    with st.expander("Update Warehouse Stock (Manual Entry)", expanded=False):
        c_add1, c_add2, c_add3, c_add4 = st.columns(4)
        target_hub = c_add1.selectbox("Select Hub", list(warehouses.keys()))
        target_item = c_add2.selectbox("Select Item", all_items)
        add_qty = c_add3.number_input("Add Quantity", min_value=1, value=100)

        if c_add4.button("Add Stock"):
            if target_item in st.session_state.warehouse_data[target_hub]["stock"]:
                st.session_state.warehouse_data[target_hub]["stock"][target_item] += add_qty
            else:
                st.session_state.warehouse_data[target_hub]["stock"][target_item] = add_qty

            st.session_state.dispatch_success = f"Stock Updated: Added {add_qty} {target_item} to {target_hub}"
            st.rerun()

    # 2. Warehouse Status Display (Fixed Loop Error)
    st.markdown("### Warehouse Status")

    # Define columns BEFORE the loop
    wc1, wc2, wc3 = st.columns(3)
    cols = [wc1, wc2, wc3]
    hubs = list(warehouses.keys())

    for i, hub in enumerate(hubs):
        # Use modulo to cycle columns if more than 3 hubs, or just limit to 3
        if i < 3:
            with cols[i]:
                st.markdown(f"#### {hub.replace('_', ' ')}")
                item_sel = st.selectbox("Item", all_items, key=f"{hub}_item")
                qty = warehouses[hub]["stock"].get(item_sel, 0)

                if qty >= 800:
                    status, color = "Enough", "green"
                elif qty >= 400:
                    status, color = "Medium", "orange"
                else:
                    status, color = "Low", "red"

                st.metric("Available Quantity", qty)
                st.markdown(f"Status: :{color}[{status}]")
                st.divider()

    # 3. Manual Dispatch (Interactive)
    st.markdown("### Emergency Dispatch Plan")
    with st.expander("Create Manual Emergency Dispatch", expanded=False):
        md1, md2, md3, md4, md5 = st.columns(5)
        m_hub = md1.selectbox("From Hub", list(warehouses.keys()), key="m_hub")
        m_dest = md2.selectbox("To District", list(DISTRICT_COORDS.keys()), key="m_dest")
        m_item = md3.selectbox("Item", all_items, key="m_item")
        m_qty = md4.number_input("Quantity", min_value=1, value=50, key="m_qty")

        if md5.button("Dispatch Truck"):
            curr = st.session_state.warehouse_data[m_hub]["stock"].get(m_item, 0)
            if curr >= m_qty:
                st.session_state.warehouse_data[m_hub]["stock"][m_item] -= m_qty
                st.session_state.manual_dispatches.append({
                    "Location": m_dest, "Item": m_item, "Quantity": m_qty,
                    "From Hub": m_hub, "Type": "MANUAL", "Status": "Dispatched"
                })
                st.session_state.dispatch_success = f"Dispatch Confirmed: {m_qty} units of {m_item} sent to {m_dest}."
                st.rerun()
            else:
                st.error(f"Dispatch Failed: Insufficient stock ({curr}).")

    # 4. Map Logic
    routes = st.session_state.get("routes")
    auto_data = pd.DataFrame()
    if routes is not None and not routes.empty:
        auto_data = routes.rename(
            columns={"Destination": "Location", "Origin_Hub": "From Hub", "Quantity": "Quantity"}).copy()
        auto_data["Type"] = "AUTO"

    if st.session_state.manual_dispatches:
        manual_df = pd.DataFrame(st.session_state.manual_dispatches)
        final_dispatch = pd.concat([auto_data, manual_df], ignore_index=True)
    else:
        final_dispatch = auto_data

    if not final_dispatch.empty:
        st.dataframe(final_dispatch[["Location", "Item", "Quantity", "From Hub", "Type", "Status"]],
                     use_container_width=True)

        # Map Visualization
        selected = st.selectbox("Select dispatch to highlight", final_dispatch.index,
                                format_func=lambda
                                    x: f"{final_dispatch.loc[x, 'From Hub']} -> {final_dispatch.loc[x, 'Location']}")

        row = final_dispatch.loc[selected]
        hub_name = row["From Hub"]
        dest_name = row["Location"]

        if hub_name in warehouses and dest_name in DISTRICT_COORDS:
            wlat = warehouses[hub_name]["lat"]
            wlon = warehouses[hub_name]["lon"]
            dlat, dlon = DISTRICT_COORDS[dest_name]

            m = folium.Map(location=[(wlat + dlat) / 2, (wlon + dlon) / 2], zoom_start=7)

            folium.Marker([wlat, wlon], tooltip=hub_name, icon=folium.Icon(color="blue", icon="home")).add_to(m)
            folium.Marker([dlat, dlon], tooltip=dest_name, icon=folium.Icon(color="red")).add_to(m)

            # Simple curved path approximation
            points = []
            steps = 25
            for i in range(steps + 1):
                t = i / steps
                lat = wlat + (dlat - wlat) * t
                lon = wlon + (dlon - wlon) * t
                lon += 0.1 * math.sin(t * math.pi)  # Curvature
                points.append((lat, lon))

            color = "orange" if row["Type"] == "MANUAL" else "green"
            if "Blocked" in str(row.get("Status", "")): color = "red"

            folium.PolyLine(points, color=color, weight=4).add_to(m)
            st_folium(m, width=900, height=500)
    else:
        st.info("No dispatch data to display.")

# ---------- About tab ----------
with tabs[6]:
    st.markdown("""
    This prototype simulates an AI-driven emergency medical supply chain.
    It forecasts demand, detects shortages, computes inventory policies,
    and plans emergency dispatch from virtual warehouses using geospatial logic.
    """)