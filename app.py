import streamlit as st
import requests
import pydeck as pdk
import math
import asyncio
import aiohttp
import pickle
from datetime import datetime, timedelta

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
MAPBOX_KEY = "pk.eyJ1IjoibGludXNkcnAiLCJhIjoiY21pN2RsajY4MDZ4ZTJqczc4cjhwYnlpMSJ9.OiKAnhY6AqXuKQt55_oLhA"
OCM_KEY = "46ec45da-3737-49e3-86bf-a677f9fe2006"

st.set_page_config(layout="wide", page_title="EV Route Planner")
# -------------------------------------------------------------
# LOAD ML MODEL
# -------------------------------------------------------------

import os
import pickle
from train_model import train_and_save_model

# falls Modell fehlt → neu trainieren
if not os.path.exists("waiting_time_model.pkl"):
    train_and_save_model()

# Modell laden
with open("waiting_time_model.pkl", "rb") as f:
    wait_model = pickle.load(f)


# -------------------------------------------------------------
# UPDATED FEATURE EXTRACTION — NOW USING REAL OCM DATA
# -------------------------------------------------------------
def extract_charger_features(charger, arrival_time, route_duration_min):

    # 1. Number of connectors (OCM data)
    num_connectors = charger.get("num_connectors", 1)

    # 2. Power (kW)
    power_kw = charger.get("power_kw")
    if power_kw is None:
        power_kw = 50  # fallback

    # 3. Charger type (AC=0, DC=1)
    charger_type = 1 if power_kw >= 50 else 0

    # 4. Hour of day
    hour = arrival_time.hour

    # 5. Day of week (0=Monday)
    weekday = arrival_time.weekday()

    # 6. Traffic factor
    expected_speed = 70
    if charger.get("proj_dist_km", 0) > 0:
        expected_duration = (charger["proj_dist_km"] / expected_speed) * 60
        traffic_factor = max(0.7, min(2.0, route_duration_min / expected_duration))
    else:
        traffic_factor = 1.0

    return [
        num_connectors,
        power_kw,
        charger_type,
        hour,
        weekday,
        traffic_factor
    ]

# -------------------------------------------------------------
# SIDEBAR INPUTS
# -------------------------------------------------------------
st.sidebar.title("Trip Inputs")
start = st.sidebar.text_input("Start", "Zurich")
dest = st.sidebar.text_input("Destination", "Munich")

st.sidebar.subheader("Driver Inputs")
manufacturer_range = st.sidebar.number_input("Manufacturer Rated Range (km)", 50, 1000, 400)
battery_soc = st.sidebar.slider("Current State of Charge (%)", 0, 100, 80)
safety_buffer_km = st.sidebar.slider("Minimum remaining range when arriving at charger (km)", 5, 80, 30)

SUBMIT = st.sidebar.button("Calculate Route")
st.title("⚡🚘 EV Route Planner")
if not SUBMIT:
    st.info("Find optimal charging stops along your route. Click the calculate button to begin.") 
    st.stop()

# -------------------------------------------------------------
# GEOCODE
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def geocode(place):
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{place}.json"
    r = requests.get(url, params={"access_token": MAPBOX_KEY})
    js = r.json()
    if js.get("features"):
        lon, lat = js["features"][0]["center"]
        return (lat, lon)
    return None

# -------------------------------------------------------------
# MAPBOX ROUTING
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_route(a, b):
    lat_a, lon_a = a
    lat_b, lon_b = b
    url = f"https://api.mapbox.com/directions/v5/mapbox/driving/{lon_a},{lat_a};{lon_b},{lat_b}"
    params = {
        "geometries": "geojson",
        "overview": "full",
        "access_token": MAPBOX_KEY,
    }
    js = requests.get(url, params=params).json()
    try:
        return js["routes"][0]["geometry"]["coordinates"]
    except:
        return []

def get_route_distance(a, b):
    route = get_route(a, b)
    if not route or len(route) < 2:
        return None
    d = 0
    for i in range(1, len(route)):
        lon1, lat1 = route[i-1]
        lon2, lat2 = route[i]
        d += haversine(lat1, lon1, lat2, lon2)
    return d

# -------------------------------------------------------------
# HAVERSINE
# -------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return 2 * R * math.asin(math.sqrt(a))

# -------------------------------------------------------------
# SAMPLE ROUTE POINTS
# This function samples n points evenly along the route
# Based from a polyline-style route (obtained from Mapbox)
# -------------------------------------------------------------
def sample_route_points(route, n=50):
    if len(route) < 2:
        return []
    cum = [0]
    for i in range(1, len(route)):
        lon1, lat1 = route[i-1]
        lon2, lat2 = route[i]
        cum.append(cum[-1] + haversine(lat1, lon1, lat2, lon2))
    total = cum[-1]
    if total == 0:
        return []
    points = []
    targets = [total * i/(n-1) for i in range(n)]
    idx = 0
    for t in targets:
        while idx < len(cum)-1 and cum[idx+1] < t:
            idx += 1
        lon, lat = route[idx]
        points.append((lat, lon))
    return points

# -------------------------------------------------------------
# OCM FETCHING
# -------------------------------------------------------------
async def fetch_ocm(session, lat, lon, radius=4):
    url = "https://api.openchargemap.io/v3/poi/"
    params = {
        "output": "json",
        "latitude": lat,
        "longitude": lon,
        "distance": radius,
        "distanceunit": "km",
        "maxresults": 200,
        "key": OCM_KEY
    }
    async with session.get(url, params=params) as resp:
        try:
            return await resp.json()
        except:
            return []

@st.cache_data(show_spinner=False)
def get_chargers_async(points):
    async def run():
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_ocm(session, lat, lon) for lat, lon in points]
            return await asyncio.gather(*tasks)
    return asyncio.run(run())

# -------------------------------------------------------------
# PARSE OCM RESULTS
# -------------------------------------------------------------
def parse_ocm_results(results):
    chargers = []
    seen = set()
    for block in results:
        if not block:
            continue

        for item in block:
            cid = item.get("ID")
            if cid in seen:
                continue
            seen.add(cid)

            addr = item.get("AddressInfo", {})
            lat = addr.get("Latitude")
            lon = addr.get("Longitude")
            if lat is None or lon is None:
                continue

            title = addr.get("Title", "Charging Station")

            parts = []
            if addr.get("AddressLine1"): parts.append(addr["AddressLine1"])
            if addr.get("Town"): parts.append(addr["Town"])
            if addr.get("Postcode"): parts.append(addr["Postcode"])
            if addr.get("Country"): parts.append(addr["Country"]["ISOCode"])
            full_addr = ", ".join(parts) if parts else "Address unavailable"

            # Extract power + connectors from OCM
            conns = item.get("Connections", [])
            if conns:
                # try to find first valid PowerKW
                power_kw = None
                for c in conns:
                    if c.get("PowerKW"):
                        power_kw = c.get("PowerKW")
                        break
                if power_kw is None:
                    power_kw = 50  # fallback
                num_con = len(conns)
            else:
                power_kw = 50
                num_con = 2

            chargers.append({
                "lat": lat,
                "lon": lon,
                "title": title,
                "address": full_addr,
                "power_kw": power_kw,
                "num_connectors": num_con,
                "icon_data": {
                    "url": "https://static.vecteezy.com/system/resources/previews/035/635/295/non_2x/ev-charging-station-map-icon-free-png.png",
                    "width": 256,
                    "height": 256,
                    "anchorY": 256
                }
            })
    return chargers

# -------------------------------------------------------------
# FILTER CHARGERS ALONG ROUTE
# -------------------------------------------------------------
def point_to_segment_distance(lat, lon, lat1, lon1, lat2, lon2):
    R = 6371
    def to_xy(a, b):
        return (math.radians(b)*R*math.cos(math.radians(a)),
                math.radians(a)*R)
    px, py = to_xy(lat, lon)
    x1, y1 = to_xy(lat1, lon1)
    x2, y2 = to_xy(lat2, lon2)
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.dist((px, py), (x1, y1))
    t = max(0, min(1, ((px-x1)*dx + (py-y1)*dy)/(dx*dx+dy*dy)))
    projx = x1 + t*dx
    projy = y1 + t*dy
    return math.dist((px, py), (projx, projy))

def filter_along_route(chargers, route, max_km=3):
    out = []
    for c in chargers:
        for i in range(len(route)-1):
            lon1, lat1 = route[i]
            lon2, lat2 = route[i+1]
            d = point_to_segment_distance(c["lat"], c["lon"], lat1, lon1, lat2, lon2)
            if d <= max_km:
                out.append(c)
                break
    return out

with st.spinner("Calculating route..."):
    # -------------------------------------------------------------
    # MAIN
    # -------------------------------------------------------------
    start_coords = geocode(start)
    dest_coords = geocode(dest)

    route_coords = get_route(start_coords, dest_coords)
    if len(route_coords) < 2:
        st.error("Route could not be calculated.")
        st.stop()

    # cumulative route km
    cum_dist = [0]
    for i in range(1, len(route_coords)):
        lon1, lat1 = route_coords[i-1]
        lon2, lat2 = route_coords[i]
        cum_dist.append(cum_dist[-1] + haversine(lat1, lon1, lat2, lon2))

    total_route_km = cum_dist[-1]

    reachable_km = max(0, manufacturer_range * (battery_soc/100) - safety_buffer_km)

    # -------------------------------------------------------------
    # GET CHARGERS
    # -------------------------------------------------------------
    sample_points = sample_route_points(route_coords, n=50)
    ocm_raw = get_chargers_async(sample_points)
    chargers = parse_ocm_results(ocm_raw)
    chargers = filter_along_route(chargers, route_coords, max_km=3)

    # -------------------------------------------------------------
    # OPTION B — CHARGER SELECTION (iterative, supports multiple stops)
    # -------------------------------------------------------------
    # Loop: repeatedly pick the best charger that is reachable after
    # the vehicle's current range. After choosing a charger, assume the car is
    # fully charged and continue from that point until destination is reachable
    # or we reach the max number of stops.
    top_chargers = []

    cur_start = start_coords
    cur_soc = battery_soc
    cumulative_km = 0.0
    max_legs = 10
    leg = 0

    while leg < max_legs:
        route_seg = get_route(cur_start, dest_coords)
        if len(route_seg) < 2:
            st.error("Could not get route for a leg — stopping itinerary planning.")
            break

        # cumulative distances for this segment
        cum = [0]
        for i in range(1, len(route_seg)):
            lon1, lat1 = route_seg[i-1]
            lon2, lat2 = route_seg[i]
            cum.append(cum[-1] + haversine(lat1, lon1, lat2, lon2))
        seg_km = cum[-1]

        reachable_km_leg = max(0, manufacturer_range * (cur_soc/100) - safety_buffer_km)

        if reachable_km_leg >= seg_km:
            # Can reach destination from cur_start
            if leg == 0 and not top_chargers:
                st.success("No charging stop required.")
            break

        # Need a charger on this leg
        sample_points_leg = sample_route_points(route_seg, n=50)
        ocm_raw_leg = get_chargers_async(sample_points_leg)
        chargers_leg = parse_ocm_results(ocm_raw_leg)
        chargers_leg = filter_along_route(chargers_leg, route_seg, max_km=3)

        # collect candidate chargers that are reachable (proj_dist <= reachable_km_leg)
        candidates = []
        for c in chargers_leg:
            mind = float("inf")
            proj = 0
            for i in range(len(route_seg)):
                lon_i, lat_i = route_seg[i]
                d = haversine(c["lat"], c["lon"], lat_i, lon_i)
                if d < mind:
                    mind = d
                    proj = i
            proj_dist = cum[proj]
            if proj_dist <= reachable_km_leg:
                candidates.append((proj_dist, c))

        if not candidates:
            st.warning("No candidate chargers found on this leg — cannot continue planning.")
            break

        # pick the candidate that is furthest along the route but still reachable
        candidates.sort(key=lambda x: x[0], reverse=True)
        chosen_proj, chosen_c = candidates[0]

        # approximate detour (same approach as before)
        dist_to_ch = get_route_distance(cur_start, (chosen_c["lat"], chosen_c["lon"]))
        dist_ch_to_end = get_route_distance((chosen_c["lat"], chosen_c["lon"]), dest_coords)
        if None in (dist_to_ch, dist_ch_to_end):
            detour = float('inf')
        else:
            # Compare against the current remaining route (cur_start -> dest), not the full original route
            # seg_km holds the length of the current segment (cur_start to dest)
            detour = dist_to_ch + dist_ch_to_end - seg_km

        # compute arrival time and predicted wait
        chosen_c["proj_dist_km"] = cumulative_km + chosen_proj
        arrival_hours = (cumulative_km + chosen_proj) / 80
        arrival_time = datetime.now() + timedelta(hours=arrival_hours)
        features = extract_charger_features(chosen_c, arrival_time, (total_route_km/80) * 60)
        predicted_wait = wait_model.predict([features])[0]
        wait_low = max(0, predicted_wait * 0.7)
        wait_high = predicted_wait * 1.3

        url = f"https://www.google.com/maps?q={chosen_c['lat']},{chosen_c['lon']}"

        st.success(
            f"**Stop {leg+1}: {chosen_c['title']}**\n"
            f"📍 **Address:** {chosen_c['address']}\n"
            f"⚡ **Power:** {chosen_c['power_kw']} kW, {chosen_c['num_connectors']} connectors\n"
            f"⏳ **Estimated waiting time:** {wait_low:.0f}–{wait_high:.0f} minutes\n"
            f"📏 **Distance along route:** {chosen_c['proj_dist_km']:.1f} km\n"
            f"↩️ **Detour (est):** {detour:.1f} km\n"
            f"[Open in Google Maps]({url})"
        )

        top_chargers.append((detour, chosen_c['proj_dist_km'], chosen_c))

        # advance to next leg starting at this charger
        cumulative_km += chosen_proj
        cur_start = (chosen_c['lat'], chosen_c['lon'])
        cur_soc = 100
        leg += 1


    # -------------------------------------------------------------
    # MAP RENDERING
    # -------------------------------------------------------------
    layers = []

    # Route (draw segments that include chosen chargers if any)
    displayed_routes = []

    if top_chargers:
        # build ordered list of waypoints: start -> chosen chargers -> destination
        waypoints = [start_coords] + [(c[2]["lat"], c[2]["lon"]) for c in top_chargers] + [dest_coords]
        for i in range(len(waypoints)-1):
            a = waypoints[i]
            b = waypoints[i+1]
            seg = get_route(a, b)
            if seg and len(seg) > 1:
                displayed_routes.append(seg)
            else:
                # fallback to a straight line between the two points (lon,lat order)
                displayed_routes.append([[a[1], a[0]], [b[1], b[0]]])
    else:
        displayed_routes = [route_coords]

    # Add each segment as a path layer
    for seg_coords in displayed_routes:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=[{"path": seg_coords}],
                get_path="path",
                get_color=[255, 0, 0],
                width_scale=20,
                width_min_pixels=3
            )
        )

    # Recommended chargers
    for det, proj_dist, c in top_chargers:
        layers.append(
            pdk.Layer(
                "IconLayer",
                data=[{
                    "lon": c["lon"],
                    "lat": c["lat"],
                    "title": c["title"],
                    "icon_data": {
                        "url": "https://cdn-icons-png.flaticon.com/512/535/535239.png",
                        "width": 256,
                        "height": 256,
                        "anchorY": 256
                    }
                }],
                get_position='[lon, lat]',
                get_icon="icon_data",
                size_scale=6,
                get_size=6,
                pickable=True
            )
        )

    # Start icon
    layers.append(
        pdk.Layer(
            "IconLayer",
            data=[{
                "lon": start_coords[1],
                "lat": start_coords[0],
                "title": "Start",
                "icon_data": {
                    "url": "https://cdn-icons-png.flaticon.com/512/684/684908.png",
                    "width": 256,
                    "height": 256,
                    "anchorY": 256
                }
            }],
            get_position='[lon, lat]',
            get_icon="icon_data",
            size_scale=5,
            get_size=10
        )
    )

    # Destination icon
    layers.append(
        pdk.Layer(
            "IconLayer",
            data=[{
                "lon": dest_coords[1],
                "lat": dest_coords[0],
                "title": "Destination",
                "icon_data": {
                    "url": "https://cdn-icons-png.flaticon.com/512/684/684908.png",
                    "width": 256,
                    "height": 256,
                    "anchorY": 256
                }
            }],
            get_position='[lon, lat]',
            get_icon="icon_data",
            size_scale=5,
            get_size=10
        )
    )

    # Center the map at the midpoint between start and destination (simple and clear)
    mid_lat = (start_coords[0] + dest_coords[0]) / 2
    mid_lon = (start_coords[1] + dest_coords[1]) / 2
    view = pdk.ViewState(
        latitude=mid_lat,
        longitude=mid_lon,
        zoom=6,
        pitch=0
    )

    deck = pdk.Deck(
        map_provider="mapbox",
        api_keys={"mapbox": MAPBOX_KEY},
        map_style="mapbox://styles/mapbox/streets-v11",
        initial_view_state=view,
        layers=layers,
        tooltip={"text": "{title}"}
    )

    st.pydeck_chart(deck)
