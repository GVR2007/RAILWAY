from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import pandas as pd
import networkx as nx
from networkx.exception import NodeNotFound
import math, datetime
import json
import threading
import time
import logging

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------- Load Rules ----------------------
try:
    rules = pd.read_csv("GSR_Rules.csv")
except pd.errors.ParserError:
    # Fallback: create rules manually if CSV parsing fails
    rules = pd.DataFrame({
        "Rule_No": ["9.01", "9.02", "9.03", "9.04", "9.05"],
        "Rule_Text": [
            "Trains must stop when the signal ahead is red.",
            "If two trains are on same line, the one behind must slow down.",
            "If head-on collision risk exists, stop both trains immediately.",
            "Maintain safe distance of at least 50 km between trains on the same route.",
            "In case of signal failure, stop all trains in the affected section."
        ]
    })

# ---------------------- Live Map Data APIs ----------------------
stations = [
    {"name": "Delhi", "lat": 28.6139, "lon": 77.2090},
    {"name": "Agra", "lat": 27.1767, "lon": 78.0081},
    {"name": "Lucknow", "lat": 26.8467, "lon": 80.9462},
    {"name": "Varanasi", "lat": 25.3176, "lon": 82.9739},
    {"name": "Kanpur", "lat": 26.4499, "lon": 80.3319},
    {"name": "Patna", "lat": 25.5941, "lon": 85.1376},
    {"name": "Jaipur", "lat": 26.9124, "lon": 75.7873},
    {"name": "Bhopal", "lat": 23.2599, "lon": 77.4126},
    {"name": "Chandigarh", "lat": 30.7333, "lon": 76.7794},
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    {"name": "Gwalior", "lat": 26.2183, "lon": 78.1828}
]

# ---------------------- Create Railway Graph ----------------------
G = nx.Graph()
edges = [
    {"source": "Delhi", "target": "Agra", "weight": 200},
    {"source": "Agra", "target": "Kanpur", "weight": 270},
    {"source": "Kanpur", "target": "Lucknow", "weight": 80},
    {"source": "Lucknow", "target": "Varanasi", "weight": 320},
    {"source": "Varanasi", "target": "Patna", "weight": 240},
    {"source": "Agra", "target": "Bhopal", "weight": 360},
    {"source": "Kanpur", "target": "Allahabad", "weight": 190},
    {"source": "Lucknow", "target": "Bareilly", "weight": 250},
    {"source": "Delhi", "target": "Jaipur", "weight": 300},
    {"source": "Jaipur", "target": "Bhopal", "weight": 400},
    {"source": "Bhopal", "target": "Mumbai", "weight": 800},
    {"source": "Delhi", "target": "Chandigarh", "weight": 250},
    {"source": "Agra", "target": "Gwalior", "weight": 120}
]
for e in edges:
    G.add_edge(e["source"], e["target"], weight=e["weight"])

# Add all stations as nodes to prevent NodeNotFound errors
for s in stations:
    G.add_node(s["name"])

# ---------------------- Helper Functions ----------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dLat, dLon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dLon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_rule(rule_no):
    row = rules[rules["Rule_No"] == rule_no]
    return row["Rule_Text"].values[0] if not row.empty else "Rule not found."

def find_best_connections(new_station, max_connections=3, max_distance_km=300):
    """Find the best stations to connect with, based on distance and connectivity."""
    lat, lon = new_station["lat"], new_station["lon"]
    scored = []

    for s in stations:
        if s["name"] == new_station["name"]:
            continue
        dist = haversine(lat, lon, s["lat"], s["lon"])
        if dist <= max_distance_km:
            degree = G.degree(s["name"])  # how many edges that station already has
            score = dist * (1 + degree * 0.2)  # penalize highly connected nodes slightly
            scored.append({"name": s["name"], "distance": dist, "score": score})

    # Sort by score (smaller = better)
    scored.sort(key=lambda x: x["score"])
    return scored[:max_connections]

def rebuild_graph():
    """Rebuilds the railway graph dynamically from current stations and edges."""
    G.clear()
    for s in stations:
        G.add_node(s["name"])
    for e in edges:
        G.add_edge(e["source"], e["target"], weight=e["weight"])
    return G

def save_network():
    with open("stations.json", "w") as f:
        json.dump(stations, f, indent=2)
    with open("edges.json", "w") as f:
        json.dump(edges, f, indent=2)

def load_network():
    global stations, edges
    try:
        with open("stations.json") as f:
            stations = json.load(f)
        with open("edges.json") as f:
            edges = json.load(f)
    except FileNotFoundError:
        pass

decision_log = []

# Global trains list (initially hardcoded, but can be added to)
trains = [
    {"name": "Shatabdi Express", "route": ["Delhi", "Agra", "Kanpur", "Lucknow"], "progress": 0.0, "speed": 0.01},
    {"name": "Rajdhani Express", "route": ["Delhi", "Jaipur", "Bhopal", "Mumbai"], "progress": 0.0, "speed": 0.008},
    {"name": "Garib Rath", "route": ["Delhi", "Chandigarh"], "progress": 0.0, "speed": 0.012}
]

# ---------------------- Route Recalculation ----------------------
def reroute_train(source, destination):
    try:
        path = nx.shortest_path(G, source=source, target=destination, weight='weight')
        return path
    except nx.NetworkXNoPath:
        return ["No alternate route available"]



# ---------- API: welcome endpoint ----------
@app.route("/welcome", methods=["GET"])
def welcome():
    app.logger.info(f"Request received: {request.method} {request.path}")
    return jsonify({"message": "Welcome to the Railway Decision API!"})

@app.route("/decide", methods=["POST"])
def decide():
    if not request.json or "trains" not in request.json:
        return jsonify({"error": "Invalid request: 'trains' key required"}), 400
    trains = request.json["trains"]
    decisions = []

    # Check pairwise distances with tiered collision avoidance
    for i, t1 in enumerate(trains):
        for j, t2 in enumerate(trains):
            if i >= j:
                continue
            try:
                lat1, lon1, lat2, lon2 = float(t1["lat"]), float(t1["lon"]), float(t2["lat"]), float(t2["lon"])
                dist = haversine(lat1, lon1, lat2, lon2)
            except (ValueError, TypeError):
                return jsonify({"error": "Invalid coordinates: lat and lon must be numbers"}), 400
            if dist < 5:  # Critical proximity
                decisions.append({
                    "action": "STOP",
                    "reason": "Critical proximity – emergency stop",
                    "affected_trains": [t1["name"], t2["name"]]
                })
            elif dist < 10:  # Moderate proximity
                decisions.append({
                    "action": "SLOW",
                    "reason": "Moderate proximity – reducing speed",
                    "affected_trains": [t1["name"], t2["name"]]
                })
            elif dist < 20:  # Reroute zone
                decisions.append({
                    "action": "REROUTE",
                    "reason": "Rerouting to avoid congestion",
                    "affected_trains": [t1["name"], t2["name"]]
                })

    if not decisions:
        decisions.append({"action": "NORMAL", "reason": "No conflict detected."})

    return jsonify(decisions)

# ---------------------- Logs API ----------------------
@app.route("/logs", methods=["GET"])
def get_logs():
    return jsonify(decision_log)

# ---------------------- Live Map Data APIs ----------------------
@app.route("/get_stations", methods=["GET"])
def get_stations():
    return jsonify(stations)

@app.route("/add_station", methods=["POST"])
def add_station():
    try:
        data = request.get_json(force=True)
        name = data.get("name")
        lat = data.get("lat")
        lon = data.get("lon")

        if not name or lat is None or lon is None:
            return jsonify({"error": "Missing station data"}), 400

        # Check if station already exists
        if name in [s["name"] for s in stations]:
            return jsonify({"error": "Station already exists"}), 400

        # Add station
        stations.append({"name": name, "lat": lat, "lon": lon})
        G.add_node(name)

        # Connect to nearest 3 existing stations dynamically
        distances = []
        for s in stations:
            if s["name"] != name:
                dist = haversine(lat, lon, s["lat"], s["lon"])
                distances.append((s["name"], dist))

        distances.sort(key=lambda x: x[1])
        nearest = distances[:3]

        for neighbor, dist in nearest:
            G.add_edge(name, neighbor, weight=dist)

        return jsonify({
            "message": f"Station '{name}' added and connected to {len(nearest)} nearby stations.",
            "connected_edges": [{"source": name, "target": n, "weight": w} for n, w in nearest]
        })

    except Exception as e:
        print("❌ Error in /add_station:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/get_trains", methods=["GET"])
def get_trains():
    return jsonify(trains)

@app.route("/get_signals", methods=["GET"])
def get_signals():
    signals = [
        {"lat": 27.5, "lon": 79.2, "state": "green"},
        {"lat": 26.7, "lon": 81.0, "state": "red"},
        {"lat": 25.3, "lon": 83.0, "state": "yellow"},
        {"lat": 26.9, "lon": 75.8, "state": "green"},
        {"lat": 23.3, "lon": 77.4, "state": "red"}
    ]
    return jsonify(signals)

@app.route("/suggest_routes", methods=["POST"])
def suggest_routes():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        source = data.get("source")
        destination = data.get("destination")

        if not source or not destination:
            return jsonify({"error": "Missing 'source' or 'destination' in request"}), 400

        # Handle case: unknown station names
        if source not in G or destination not in G:
            # Try to find nearest known nodes (fallback)
            def find_nearest_station(name):
                if name not in [s["name"] for s in stations]:
                    # Try approximate match (e.g. "new delhi" vs "Delhi")
                    matches = [s["name"] for s in stations if name.lower() in s["name"].lower()]
                    if matches:
                        return matches[0]
                return None

            nearest_source = find_nearest_station(source)
            nearest_dest = find_nearest_station(destination)

            if not nearest_source or not nearest_dest:
                return jsonify({
                    "error": f"Unknown station(s): {source if source not in G else ''} {destination if destination not in G else ''}",
                    "suggestion": "Ensure both stations exist or add them using /add_station"
                }), 404

            source = nearest_source
            destination = nearest_dest

        # Compute multiple route suggestions using all simple paths with cutoff
        if not nx.has_path(G, source, destination):
            return jsonify({"error": f"No valid path found between {source} and {destination}"}), 404

        from networkx.algorithms.simple_paths import all_simple_paths
        all_paths = list(all_simple_paths(G, source, destination, cutoff=10))  # Limit to avoid excessive computation

        # Calculate distances for each path
        path_distances = []
        for path in all_paths:
            total_dist = 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                total_dist += G[u][v]['weight']
            path_distances.append((path, total_dist))

        # Sort by distance and select top 3
        path_distances.sort(key=lambda x: x[1])
        routes = [{"path": path, "distance": dist} for path, dist in path_distances[:3]]

        return jsonify({
            "routes": routes,
            "message": f"Found {len(routes)} possible route(s) between {source} and {destination}",
            "source": source,
            "destination": destination
        })

    except Exception as e:
        print("❌ suggest_routes Error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/get_edges", methods=["GET"])
def get_edges():
    return jsonify(edges)

@app.route("/add_edge", methods=["POST"])
def add_edge():
    data = request.json
    source = data.get("source")
    target = data.get("target")
    weight = data.get("weight", 100)

    if not source or not target:
        return jsonify({"error": "Source and target required"}), 400

    # Prevent duplicate edges
    if any(e["source"] == source and e["target"] == target for e in edges):
        return jsonify({"error": "Edge already exists"}), 400

    edges.append({"source": source, "target": target, "weight": int(weight)})

    # Rebuild the graph dynamically
    rebuild_graph()
    save_network()
    return jsonify({"message": "Edge added successfully"}), 200

@app.route("/add_train", methods=["POST"])
def add_train():
    data = request.json
    name = data.get("name")
    source = data.get("source")
    destination = data.get("destination")
    speed = data.get("speed", 60)
    route = data.get("route")
    if not name or not source or not destination:
        return jsonify({"error": "Name, source, destination required"}), 400
    if not route:
        # Generate route if not provided
        try:
            route = nx.shortest_path(G, source=source, target=destination, weight='weight')
        except nx.NetworkXNoPath:
            return jsonify({"error": "No route available"}), 400
    train = {
        "name": name,
        "route": route,
        "progress": 0.0,
        "speed": float(speed) / 10000  # Convert km/h to simulation speed
    }
    trains.append(train)
    return jsonify({"message": "Train added successfully"})

@app.route("/rules", methods=["GET"])
def get_rules():
    return jsonify(rules.to_dict(orient='records'))

@app.route("/analytics", methods=["GET"])
def get_analytics():
    total_decisions = len(decision_log)
    avg_delay = 0  # Placeholder
    rule_frequency = {}
    for log in decision_log:
        rule = log.get("reason", "").split(" – ")[0]
        rule_frequency[rule] = rule_frequency.get(rule, 0) + 1
    return jsonify({
        "total_decisions": total_decisions,
        "avg_delay": avg_delay,
        "rule_frequency": rule_frequency
    })

@app.route("/weather/<city>", methods=["GET"])
def get_weather(city):
    # Mock weather data
    weather_data = {
        "Delhi": {"city": "Delhi", "condition": "Sunny, 32°C"},
        "Agra": {"city": "Agra", "condition": "Clear, 30°C"},
        "Lucknow": {"city": "Lucknow", "condition": "Cloudy, 28°C"},
        "Varanasi": {"city": "Varanasi", "condition": "Rainy, 26°C"}
    }
    return jsonify(weather_data.get(city, {"city": city, "condition": "Unknown"}))

@app.route("/get_dashboard_stats", methods=["GET"])
def get_dashboard_stats():
    try:
        total_stations = len(stations)
        total_trains = len(trains)
        total_alerts = len(decision_log)
        total_maintenance = 1  # Placeholder for maintenance tracker

        total_revenue = sum([t.get("revenue", 0) for t in trains])
        schedule_accuracy = 95  # Placeholder

        return jsonify({
            "stations": total_stations,
            "trains": total_trains,
            "alerts": total_alerts,
            "maintenance": total_maintenance,
            "schedule_accuracy": schedule_accuracy,
            "revenue": total_revenue
        })
    except Exception as e:
        print("Error in /get_dashboard_stats:", e)
        return jsonify({"error": str(e)}), 500

# ---------------------- Background Simulation Thread ----------------------
def simulation_thread():
    """Background thread to run simulation loop, updating trains and revenue."""
    while True:
        for train in trains:
            # Simulate progress and revenue increase
            train["progress"] += train["speed"]
            if "revenue" not in train:
                train["revenue"] = 0
            train["revenue"] += 500  # Simulate revenue growth
            if train["progress"] >= 1.0:
                train["progress"] = 0.0  # Loop back
        time.sleep(2)  # Update every 2 seconds



# Start simulation thread
threading.Thread(target=simulation_thread, daemon=True).start()

if __name__ == "__main__":
    app.run(port=5000)
