import heapq
import googlemaps
import requests
import math
from pprint import pprint
from util import PriorityQueue, Stack
from geopy.distance import geodesic
import networkx as nx
import folium
import argparse

# Initialize Google Maps API client
google_api_file = open("google_api_key.txt", "r")
google_api_key = google_api_file.read()
google_api_file.close()

gmaps = googlemaps.Client(key=google_api_key)

# Initialize NREL API key
nrel_api_file = open("nrel_api_key.txt", "r")
nrel_api_key = nrel_api_file.read()
nrel_api_file.close()


class Constants:
    MAX_RANGE = 400  # mi
    THRESHOLD = 20  # mi
    DIST_FROM_STATION = 10  # mi
    GOAL_RADIUS = 0.2  # mi
    START = ""
    DESTINATION = ""
    WAYPOINTS = []
    STATIONS = []
    ROUTE_GRAPH = None


class Node:
    def __init__(self, state, path, cost=0):
        self.state = state
        self.path = path
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost


# Return straight-line distance in miles between two locations
def get_distance(start, end):
    # Request directions
    return geodesic(start, end).miles


# Get the name of the node in ROUTE_GRAPH from a (lat, lng) pair
def get_node_from_pos(pos):
    route_data = dict(Constants.ROUTE_GRAPH.nodes.data("pos"))
    return list(route_data.keys())[list(route_data.values()).index(pos)]


# Return list of waypoints between two locations
def get_waypoints_along_route(current_location, destination):
    # Request directions
    directions_result = gmaps.directions(current_location, destination, mode="driving")

    # Extract waypoints
    steps = directions_result[0]["legs"][0]["steps"]

    waypoints = [
        (
            step["end_location"]["lat"],
            step["end_location"]["lng"],
        )
        for step in steps
    ]

    return waypoints


# Return list of charging stations within a given range of a route
def get_stations_along_route(start, end):
    url = f"https://developer.nrel.gov/api/alt-fuel-stations/v1/nearby-route.json"

    directions_result = gmaps.directions(start, end, mode="driving")

    # Extract the polyline points from the directions result
    polyline = directions_result[0]["overview_polyline"]["points"]

    # Decode polyline into lat, lng coordinates
    lat_lng = googlemaps.convert.decode_polyline(polyline)

    # Format as WKT LINESTRING
    linestring = (
        "LINESTRING("
        + ", ".join([f"{point['lng']} {point['lat']}" for point in lat_lng])
        + ")"
    )

    data = {
        "route": linestring,
        "distance": Constants.DIST_FROM_STATION,
        "fuel_type": "ELEC",
        "ev_charging_level": "dc_fast",
        "status": "E",
        "access": "public",
    }

    headers = {"X-Api-Key": nrel_api_key}

    # Make API request
    response = requests.post(url, data=data, headers=headers)

    if response.status_code == 200:
        data = response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return

    # Check if 'fuel_stations' key exists in the response
    if "fuel_stations" in data and data["fuel_stations"]:
        stations = [
            (
                station["latitude"],
                station["longitude"],
            )
            for station in data["fuel_stations"]
        ]
        return stations
    else:
        # Handle cases where 'fuel_stations' is not in the response
        raise Exception("No fuel stations found along route.")


def generate_route_graph():
    G = nx.DiGraph()

    for i, waypoint in enumerate(Constants.WAYPOINTS):
        G.add_node(f"waypoint_{i}", pos=waypoint)

    for i, station in enumerate(Constants.STATIONS):
        G.add_node(f"station_{i}", pos=station)

    for i in range(len(Constants.WAYPOINTS) - 1):
        G.add_edge(f"waypoint_{i}", f"waypoint_{i + 1}")

    waypoint_nodes = [node for node in G.nodes if node.startswith("waypoint")]

    for station_node in G.nodes:
        if station_node.startswith("station"):
            station_pos = G.nodes[station_node]["pos"]
            nearest_waypoints = sorted(
                waypoint_nodes,
                key=lambda node: get_distance(G.nodes[node]["pos"], station_pos),
            )[:2]

            for waypoint_node in nearest_waypoints:
                G.add_edge(station_node, waypoint_node)
                G.add_edge(waypoint_node, station_node)

    return G


# Return a list of successors in the form (action, state, step_cost)
def successor(state):
    current_location, current_range = state
    successors = []

    current_node = get_node_from_pos(current_location)

    adjacent_nodes = Constants.ROUTE_GRAPH[current_node]

    for node in adjacent_nodes:
        # Ensure node is within range
        if (
            get_distance(current_location, Constants.ROUTE_GRAPH.nodes[node]["pos"])
            >= current_range - Constants.THRESHOLD
        ):
            continue

        # Next waypoint from this one
        if node.startswith("waypoint"):
            waypoint = Constants.ROUTE_GRAPH.nodes[node]["pos"]
            distance_to_waypoint = get_distance(current_location, waypoint)
            new_range = current_range - distance_to_waypoint
            next_state = (waypoint, new_range)
            next_action = ("Waypoint", distance_to_waypoint, waypoint)
            successors.append((next_action, next_state, distance_to_waypoint))

        # Nearby stations
        elif node.startswith("station"):
            station = Constants.ROUTE_GRAPH.nodes[node]["pos"]
            distance_to_station = get_distance(current_location, station)
            new_range = Constants.MAX_RANGE
            next_state = (station, new_range)
            next_action = ("Station", distance_to_station, station)
            successors.append((next_action, next_state, distance_to_station))

    return successors


def goal_test(state):
    current_location, current_range = state

    return (
        get_distance(current_location, Constants.DESTINATION)
        <= Constants.DIST_FROM_GOAL
        and current_range >= Constants.THRESHOLD
    )


def heuristic(state):
    current_location, current_range = state

    dist = get_distance(current_location, Constants.DESTINATION)

    if dist == 0:
        return 0

    heuristic = dist / (1 + math.exp((math.log(dist) * (current_range - dist)) / dist))

    return heuristic


def dfs(initial_state, successor, goal_test):
    initial_node = Node(initial_state, [], 0)
    frontier = Stack()
    frontier.push(initial_node)
    explored = set()

    while not frontier.isEmpty():
        curr_node: Node = frontier.pop()
        explored.add(curr_node.state)

        if goal_test(curr_node.state):
            print("Goal!")
            print(f"Current state: {curr_node.state}")
            print(f"Current path:\n")
            pprint(curr_node.path)
            return curr_node.path

        for action, next_state, _ in successor(curr_node.state):
            if next_state not in explored:
                state_list = [node.state for node in frontier.list]
                if next_state in state_list:
                    frontier.list.remove(frontier.list[state_list.index(next_state)])

                next_node = Node(next_state, curr_node.path + [action])
                frontier.push(next_node)


# Uniform Cost Search
def ucs(initial_state, successor, goal_test):
    initial_node = Node(initial_state, [], 0)
    frontier = PriorityQueue()
    frontier.push(initial_node, 0)
    explored = set()

    while not frontier.isEmpty():
        curr_node: Node = frontier.pop()
        curr_state = curr_node.state
        explored.add(curr_state)

        print(curr_state)

        if goal_test(curr_state):
            print("Goal!")
            print(f"Current state: {curr_node.state}")
            print(f"Current path: {curr_node.path}")
            return curr_node.path

        for action, next_state, step_cost in successor(curr_node.state):
            next_cost = curr_node.cost + step_cost
            next_node = Node(next_state, curr_node.path + [action], next_cost)

            if next_state not in explored:
                found = False
                existing_priority = 0
                # Check for node with next_state in the frontier
                for priority, _, item in frontier.heap:
                    if item.state == next_state:  # maybe not item.state
                        found = True
                        existing_priority = priority
                        break

                # If frontier does not contain node with next_state:
                if not found:
                    frontier.push(next_node, next_cost)
                # If next_state's node in frontier has priority > next_cost:
                elif next_cost < existing_priority:
                    frontier.update(next_node, next_cost)


# Weighted A* Search
def a_star(initial_state, successor, goal_test, heuristic):
    initial_node = Node(initial_state, [], 0)
    frontier = [(initial_node, heuristic(initial_state))]
    explored = []

    while frontier:
        curr_node, _ = heapq.heappop(frontier)
        print(
            f"Exploring node: {curr_node.state} with cost: {curr_node.cost} and heuristic: {heuristic(curr_node.state)}"
        )
        explored.append(curr_node.state)

        if goal_test(curr_node.state):
            print("Goal!")
            print(f"Current state: {curr_node.state}")
            print(f"Current path:\n")
            pprint(curr_node.path)
            return curr_node.path

        for action, next_state, step_cost in successor(curr_node.state):
            next_cost = curr_node.cost + step_cost
            next_node = Node(next_state, curr_node.path + [action], next_cost)
            # print(
            #     f"Adding successor: {next_state} with action: {action} and step cost: {step_cost}"
            # )
            next_priority = next_cost + heuristic(next_state)  # * W

            if next_state not in explored and all(
                next_state != node.state for node, _ in frontier
            ):
                heapq.heappush(frontier, (next_node, next_priority))
            else:
                for i, (node, priority) in enumerate(frontier):
                    if node.state == next_state and priority > next_priority:
                        frontier[i] = (next_node, next_priority)
                        heapq.heapify(frontier)
                        break


def plot_path(path: list):
    route_start = path[0][2]
    route_map = folium.Map(location=route_start, zoom_start=10)

    folium.Marker(
        location=Constants.START,
        popup="Start",
        tooltip="Start",
        icon=folium.Icon(color="blue"),
    ).add_to(route_map)

    folium.Marker(
        location=Constants.DESTINATION,
        popup="Destination",
        tooltip="Destination",
        icon=folium.Icon(color="green"),
    ).add_to(route_map)

    for i, action in enumerate(path):
        if action[0] == "Waypoint":
            folium.Marker(
                location=action[2],
                popup=action[2],
                tooltip=f"Step {i + 1}: Waypoint",
                icon=folium.Icon(color="black"),
            ).add_to(route_map)
        elif action[0] == "Station":
            folium.Marker(
                location=action[2],
                popup=action[2],
                tooltip=f"Step {i + 1}: Station",
                icon=folium.Icon(color="red"),
            ).add_to(route_map)

    route_map.save(f"routes/{start_str}_{dest_str}.html")


if __name__ == "__main__":
    DEFAULT = dict(
        max_range=400, initial_range=400, threshold=50, station_dist=10, goal_dist=0.2
    )

    parser = argparse.ArgumentParser(
        prog="EV Route Finder",
        description="Finds the quickest route between two locations, optimized for EVs.",
    )

    parser.add_argument(
        "-s",
        "--start",
        help="Start location for this route.",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--dest",
        help="Destination location for this route.",
        required=True,
    )
    parser.add_argument(
        "--max-range",
        help="Maximum range (mi) on a full charge.",
        required=False,
        default=DEFAULT["max_range"],
        type=int,
    )
    parser.add_argument(
        "--initial-range",
        help="Initial range (mi) of the vehicle.",
        required=False,
        default=DEFAULT["initial_range"],
        type=int,
    )
    parser.add_argument(
        "--threshold",
        help="Minimum range (mi) to maintain at all times.",
        required=False,
        default=DEFAULT["threshold"],
        type=int,
    )
    parser.add_argument(
        "--station-dist",
        help="Maximum distance (mi) from route to find charging stations.",
        required=False,
        default=DEFAULT["station_dist"],
        type=int,
    )
    parser.add_argument(
        "--goal-dist",
        help="Maximum allowed distance (mi) from the destination to end route.",
        required=False,
        default=DEFAULT["goal_dist"],
        type=int,
    )

    args = parser.parse_args()

    Constants.MAX_RANGE = args.max_range
    Constants.THRESHOLD = args.threshold
    Constants.DIST_FROM_STATION = args.station_dist
    Constants.DIST_FROM_GOAL = args.goal_dist

    start_str = args.start
    start_geocode_result = gmaps.geocode(start_str)[0]["geometry"]["location"]
    Constants.START = (
        start_geocode_result["lat"],
        start_geocode_result["lng"],
    )

    dest_str = args.dest
    dest_geocode_result = gmaps.geocode(dest_str)[0]["geometry"]["location"]
    Constants.DESTINATION = (
        dest_geocode_result["lat"],
        dest_geocode_result["lng"],
    )

    initial_state = (Constants.START, 400)

    Constants.WAYPOINTS = get_waypoints_along_route(
        Constants.START, Constants.DESTINATION
    )
    Constants.WAYPOINTS.insert(0, Constants.START)
    Constants.WAYPOINTS.append(Constants.DESTINATION)

    Constants.STATIONS = get_stations_along_route(
        Constants.START, Constants.DESTINATION
    )

    Constants.ROUTE_GRAPH = generate_route_graph()

    final_path = a_star(initial_state, successor, goal_test, heuristic)

    plot_path(final_path)


# # Logic for optimal path
# def successor1(state):
#     current_location, current_range = state
#     successors = []
#     last_waypoint_reached = None

#     waypoints = get_adjacent_waypoints(current_location, destination)
#     for i, waypoint in enumerate(waypoints):
#         distance_to_next_waypoint = get_distance(current_location, waypoint)

#         # Proceed to next waypoint if enough range
#         if current_range >= distance_to_next_waypoint:
#             new_range = current_range - distance_to_next_waypoint
#             action = f"Waypoint {i}"
#             next_state = (waypoint, new_range)
#             step_cost = distance_to_next_waypoint
#             successors.append((action, next_state, step_cost))
#             last_waypoint_reached = waypoint
#             current_location = waypoint
#             current_range = new_range

#         # Charge if range is not sufficient
#         if current_range < distance_to_next_waypoint + THRESHOLD:
#             nearest_station = min(STATIONS, key=lambda station: get_distance(current_location, station))
#             distance_to_station = get_distance(current_location, nearest_station)
#             if current_range >= distance_to_station:
#                 action = f"Charging {i}"
#                 next_state = (nearest_station, NEW_RANGE)
#                 step_cost = distance_to_station
#                 successors.append((action, next_state, step_cost))
#                 current_location = last_waypoint_reached
#                 current_range = NEW_RANGE

#     return successors

# print(successor1(initial_state))
