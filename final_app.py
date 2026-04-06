import streamlit as st
import pandas as pd
import folium
import joblib
import requests
import numpy as np
import plotly.express as px
import osmnx as ox
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster


st.set_page_config(layout="wide", page_title="EnviroScan", page_icon="🌐")

WEATHER_KEY = "add_your_API_key"
OPENAQ_KEY = "add_your_API_key"


# LOAD DATA
@st.cache_resource
def load():
    df = pd.read_csv("enviro_scan_dataset.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    rf = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")

    return df, rf, scaler, le

df, rf_model, scaler, le = load()


# WEATHER API
def get_weather(lat, lon):
    try:
        res = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "appid": WEATHER_KEY, "units": "metric"},
            timeout=5
        ).json()

        return res["main"]["temp"], res["main"]["humidity"], res["wind"]["speed"]
    except:
        return 30, 60, 3


# POLLUTION API
def get_pollution(lat, lon):
    try:
        headers = {"X-API-Key": OPENAQ_KEY}

        res = requests.get(
            "https://api.openaq.org/v3/latest",
            headers=headers,
            params={"coordinates": f"{lat},{lon}", "radius": 25000, "limit": 1},
            timeout=5
        ).json()

        if res.get("results"):
            measurements = res["results"][0]["measurements"]

            data = {"pm25":40,"pm10":70,"no2":25,"so2":10,"co":1,"o3":20}

            for m in measurements:
                param = m["parameter"].lower()
                if param in data:
                    data[param] = m["value"]

            return data, "OpenAQ"
    except:
        pass

    try:
        res = requests.get(
            "https://api.openweathermap.org/data/2.5/air_pollution",
            params={"lat": lat, "lon": lon, "appid": WEATHER_KEY},
            timeout=5
        ).json()

        comp = res["list"][0]["components"]

        return {
            "pm25": comp.get("pm2_5", 40),
            "pm10": comp.get("pm10", 70),
            "no2": comp.get("no2", 25),
            "so2": comp.get("so2", 10),
            "co": comp.get("co", 1),
            "o3": comp.get("o3", 20)
        }, "OpenWeather"
    except:
        return {"pm25":40,"pm10":70,"no2":25,"so2":10,"co":1,"o3":20}, "Default"

# GEO FEATURES (REAL DATA)
def get_osm_features(lat, lon):
    try:
        G = ox.graph_from_point((lat, lon), dist=1000)
        road = len(G.nodes)

        tags = {"landuse": ["industrial", "farmland"]}
        gdf = ox.features_from_point((lat, lon), tags=tags, dist=1000)

        industry = len(gdf[gdf["landuse"] == "industrial"]) if "landuse" in gdf else 0
        farmland = len(gdf[gdf["landuse"] == "farmland"]) if "landuse" in gdf else 0

        dump = 0

        return road, industry, farmland, dump

    except:
        return 0, 0, 0, 0


# MENU
menu = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard", "Source Detection", "Health Audit", "Dataset Explorer"]
)


# DASHBOARD
if menu == "Dashboard":

    st.title("Pollution Dashboard")

    with st.expander(" Filters"):
        col1, col2 = st.columns(2)

        with col1:
            place = st.multiselect("Location", df["place"].unique(), df["place"].unique())

        with col2:
            source_filter = st.multiselect("Source", df["pollution_source"].unique(), df["pollution_source"].unique())

    filtered_df = df[
        (df["place"].isin(place)) &
        (df["pollution_source"].isin(source_filter))
    ]

    if filtered_df.empty:
        st.warning("No data available")
        st.stop()

    map_df = filtered_df.dropna(subset=["latitude", "longitude"])

    m = folium.Map(
        location=[map_df["latitude"].mean(), map_df["longitude"].mean()],
        zoom_start=5
    )

    HeatMap(map_df[["latitude","longitude","pm25"]].values.tolist()).add_to(m)

    cluster = MarkerCluster().add_to(m)

    def get_color(pm25):
        if pm25 <= 50:
            return "green"
        elif pm25 <= 100:
            return "orange"
        else:
            return "red"

    for _, row in map_df.iterrows():
        popup = f"""
        <b>{row['place']}</b><br>
        PM2.5: {row['pm25']}<br>
        PM10: {row['pm10']}<br>
        NO2: {row['no2']}<br>
        SO2: {row['so2']}<br>
        CO: {row['co']}<br>
        O3: {row['o3']}
        """

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            color=get_color(row["pm25"]),
            fill=True,
            tooltip=f"{row['place']} | PM2.5: {round(row['pm25'],2)}",
            popup=popup
        ).add_to(cluster)

    st_folium(m, use_container_width=True, height=500)
   
#PIE CHART
    st.subheader("Pollution Source Distribution")
    pie = filtered_df.groupby("pollution_source")["pm25"].mean().reset_index()
    fig = px.pie(
        pie,
        names="pollution_source",
        values="pm25",
        title="Average PM2.5 Contribution by Source"
    )
    st.plotly_chart(fig, use_container_width=True)


# SOURCE DETECTION
elif menu == "Source Detection":
    st.title(" Source Detection")
    lat = st.number_input("Latitude", value=12.97)
    lon = st.number_input("Longitude", value=77.59)
    if st.button("Run"):
        temp, hum, wind = get_weather(lat, lon)
        pol, src = get_pollution(lat, lon)

        # REAL GEO FEATURES
        road, ind, farm, dump = get_osm_features(lat, lon)

        st.success(f" Source: {src}")

        cols = st.columns(6)
        for i, k in enumerate(["pm25","pm10","no2","so2","co","o3"]):
            cols[i].metric(k.upper(), round(pol[k],2))

        # SHOW GEO FEATURES
        st.subheader(" Area Features")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Road Count", road)
        c2.metric("Industry Count", ind)
        c3.metric("Farmland Count", farm)
        c4.metric("Dump Count", dump)

        try:
            input_df = pd.DataFrame([{
                "pm25":pol["pm25"],
                "pm10":pol["pm10"],
                "no2":pol["no2"],
                "so2":pol["so2"],
                "co":pol["co"],
                "o3":pol["o3"],
                "temperature":temp,
                "humidity":hum,
                "wind_speed":wind,
                "road_count":road,
                "industry_count":ind,
                "farmland_count":farm,
                "dump_count":dump,
                "pollution_index":pol["pm25"]+pol["pm10"]+pol["no2"],
                "gas_ratio":pol["no2"]/(pol["co"]+1),
                "pm_ratio":pol["pm25"]/(pol["pm10"]+1)
            }])

            required = list(scaler.feature_names_in_)

            for col in required:
                if col not in input_df.columns:
                    input_df[col] = 0

            input_df = input_df[required]

            pred = rf_model.predict(scaler.transform(input_df))
            result = le.inverse_transform(pred)[0]

            st.success(f" Predicted Source: {result}")

        except:
            st.warning("Prediction temporarily unavailable")

# HEALTH
elif menu == "Health Audit":

    st.title(" Health Audit")

    alerts = df[df["pm25"] > 130]

    if not alerts.empty:
        st.error("High Pollution Detected")
        st.dataframe(alerts)
    else:
        st.success(" Safe Air")


# DATASET
elif menu == "Dataset Explorer":

    st.title(" Dataset")

    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(index=False), "dataset.csv")