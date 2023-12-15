import heapq
import json
import webbrowser
import googlemaps
import requests
import math
from pprint import pprint
from util import PriorityQueue, Stack, Queue
from geopy.distance import geodesic
import networkx as nx
import folium
import argparse
import pandas as pd
import cProfile

# Initialize Google Maps API client
google_api_file = open("google_api_key.txt", "r")
google_api_key = google_api_file.read()
google_api_file.close()

gmaps = googlemaps.Client(key=google_api_key)

# Initialize NREL API key
nrel_api_file = open("nrel_api_key.txt", "r")
nrel_api_key = nrel_api_file.read()
nrel_api_file.close()

DEST_DIST_CACHE = {}
START_DIST_CACHE = {}


class Constants:
    """
    Class for storing various constants.
    """

    MAX_RANGE = 300  # mi
    MIN_THRESHOLD = 0.1 * MAX_RANGE  # mi, 10% of max range
    MAX_THRESHOLD = 0.8 * MAX_RANGE  # mi, 80% of max range
    TIME_TO_CHARGE = 25  # min, from 10% to 80%
    CHARGE_RATE = 0  # mi / min
    DIST_FROM_STATION = 10  # mi
    DIST_FROM_GOAL = 0.2  # mi
    AVERAGE_SPEED = 0  # mph
    START = None
    DESTINATION = None
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
    """
    Return the straight-line distance (in miles) between two locations.

    Params
    ------
    `start`: (int, int)
        Initial location as a (latitude, longitude) pair
    `end`: (int, int)
        Final location as a (latitude, longitude) pair

    Returns
    -------
    Straight-line distance (in miles) between `start` and `end`
    """

    return geodesic(start, end).miles


def dist_from_start(location):
    return get_distance(Constants.START, location)

    start_dist = START_DIST_CACHE.get(location)

    # If this location is not in the cache, calculate it and add it
    if start_dist is None:
        dist = get_distance(Constants.START, location)
        START_DIST_CACHE[location] = dist
        return dist
    else:
        return start_dist


def dist_from_dest(location):
    dest_dist = DEST_DIST_CACHE.get(location)

    # If this location is not in the cache, calculate it and add it
    if dest_dist is not None:
        return dest_dist
    else:
        dist = get_distance(Constants.DESTINATION, location)
        DEST_DIST_CACHE[location] = dist
        return dist


# Get the estimated duration (in minutes) to travel between two locations
def get_duration(start, end):
    return get_distance(start, end) / Constants.AVERAGE_SPEED * 60


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


def get_vehicles():
    url = f"https://developer.nrel.gov/api/vehicles/v1/light_duty_automobiles.json"

    params = {"api_key": nrel_api_key, "fuel_id": 41}  # ID for electric vehicles

    # Make API request
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data: dict = response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return

    if data.get("result", None) == None:
        print(f"Error: no result in data.")
        return
    else:
        df = pd.json_normalize(data["result"])

    # df.to_csv("ev_info.csv")

    df[
        [
            "model_year",
            "manufacturer_name",
            "model",
        ]
    ]

    return df


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
    station_nodes = sorted(
        [node for node in G.nodes if node.startswith("station")],
        key=lambda node: dist_from_dest(G.nodes[node]["pos"]),
        reverse=True,
    )

    # Connect each station node to adjacent waypoints and future stations
    for i in range(len(station_nodes) - 1):
        station_node = station_nodes[i]
        station_pos = G.nodes[station_node]["pos"]

        waypoints_ahead = []
        waypoints_behind = []

        for waypoint_node in waypoint_nodes:
            waypoint_pos = G.nodes[waypoint_node]["pos"]
            waypoint_start_dist = dist_from_start(waypoint_pos)
            station_start_dist = dist_from_start(station_pos)
            if waypoint_start_dist > station_start_dist:
                waypoints_ahead.append((waypoint_node, waypoint_pos))
            elif waypoint_start_dist < station_start_dist:
                waypoints_behind.append((waypoint_node, waypoint_pos))

        if waypoints_behind:
            nearest_waypoint_behind = min(
                waypoints_behind,
                key=lambda waypoint: get_distance(station_pos, waypoint[1]),
            )[0]
            G.add_edge(nearest_waypoint_behind, station_node)

        if waypoints_ahead:
            nearest_waypoint_ahead = min(
                waypoints_ahead,
                key=lambda waypoint: get_distance(station_pos, waypoint[1]),
            )[0]
            G.add_edge(station_node, nearest_waypoint_ahead)

        next_station_nodes = station_nodes[i:]

        for next_station_node in next_station_nodes:
            next_station_pos = G.nodes[next_station_node]["pos"]
            if (
                get_distance(station_pos, next_station_pos)
                < Constants.MAX_THRESHOLD - Constants.MIN_THRESHOLD
            ):
                G.add_edge(station_node, next_station_node)

    return G


# Return a list of successors in the form (action, state, step_cost)
def successor(state):
    current_location, current_range = state
    successors = []

    current_node = get_node_from_pos(current_location)

    adjacent_nodes = Constants.ROUTE_GRAPH[current_node]

    for node in adjacent_nodes:
        distance_to_node = get_distance(
            current_location, Constants.ROUTE_GRAPH.nodes[node]["pos"]
        )
        # Ensure node is within range
        if distance_to_node >= current_range - Constants.MIN_THRESHOLD:
            continue

        # Next waypoint from this one
        if node.startswith("waypoint"):
            waypoint_pos = Constants.ROUTE_GRAPH.nodes[node]["pos"]
            # distance_to_waypoint = get_distance(current_location, waypoint)

            time_to_waypoint = distance_to_node / Constants.AVERAGE_SPEED * 60

            new_range = current_range - distance_to_node
            next_state = (waypoint_pos, new_range)

            next_action = ("Waypoint", time_to_waypoint, waypoint_pos)
            successors.append((next_action, next_state, time_to_waypoint))

            # next_action = ("Waypoint", distance_to_waypoint, waypoint)
            # successors.append((next_action, next_state, distance_to_waypoint))

        # Nearby stations
        elif node.startswith("station"):
            station_pos = Constants.ROUTE_GRAPH.nodes[node]["pos"]
            # distance_to_station = get_distance(current_location, station)

            new_range = current_range - distance_to_node
            time_to_station = distance_to_node / Constants.AVERAGE_SPEED * 60
            if new_range >= Constants.MAX_THRESHOLD:
                charge_time = 0
            else:
                charge_time = (
                    Constants.MAX_THRESHOLD - new_range
                ) / Constants.CHARGE_RATE
                new_range = Constants.MAX_THRESHOLD
            total_time = charge_time + time_to_station
            next_state = (station_pos, new_range)
            next_action = ("Station", charge_time, station_pos)
            successors.append((next_action, next_state, total_time))

            # new_range = Constants.MAX_THRESHOLD
            # next_state = (station, new_range)
            # next_action = ("Station", distance_to_station, station)
            # successors.append((next_action, next_state, distance_to_station))

    return successors


def goal_test(state):
    current_location, current_range = state

    return (
        dist_from_dest(current_location) <= Constants.DIST_FROM_GOAL
        and current_range >= Constants.MIN_THRESHOLD
    )


def heuristic(state):
    current_location, current_range = state

    dist = dist_from_dest(current_location)
    if dist == 0:
        return 0

    heuristic = dist / (1 + math.exp((math.log(dist) * (current_range - dist)) / dist))

    # time_to_dist = get_duration(current_location, Constants.DESTINATION)
    # range_as_time = current_range / Constants.AVERAGE_SPEED * 60
    # max_threshold_dist = Constants.MAX_THRESHOLD / Constants.AVERAGE_SPEED * 60

    # if time_to_dist == 0:
    #     return 0

    # heuristic = time_to_dist / (
    #     1
    #     + math.exp(
    #         (math.log(time_to_dist) * (range_as_time - (max_threshold_dist / 2)))
    #         / time_to_dist
    #     )
    # )

    return heuristic


# Breadth First Search
def bfs(initial_state, successor, goal_test):
    initial_node = Node(initial_state, [])
    frontier = Queue()
    frontier.push(initial_node)
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

        for action, next_state, _ in successor(curr_node.state):
            next_node = Node(next_state, curr_node.path + [action])

            if next_state not in explored:
                found = False
                # Check for node with next_state in the frontier
                for item in frontier.list:
                    if item.state == next_state:  # maybe not item.state
                        found = True
                        break

                # If frontier does not contain node with next_state:
                if not found:
                    frontier.push(next_node)


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


def a_star(initial_state, successor, goal_test, heuristic):
    initial_node = Node(initial_state, [], 0)
    frontier = PriorityQueue()
    frontier.push(initial_node, 0)
    explored = set()

    while not frontier.isEmpty():
        curr_node = frontier.pop()
        print(
            f"Exploring node: {curr_node.state} with cost: {curr_node.cost} and heuristic: {heuristic(curr_node.state)}"
        )
        explored.add(curr_node.state)

        if goal_test(curr_node.state):
            print("Goal!")
            print(f"Current state: {curr_node.state}")
            print(f"Current path:\n")
            pprint(curr_node.path)
            return curr_node.path

        for action, next_state, step_cost in successor(curr_node.state):
            next_cost = curr_node.cost + step_cost
            next_node = Node(next_state, curr_node.path + [action], next_cost)
            next_priority = next_cost + heuristic(next_state)

            if next_state not in explored:
                state_indices = {
                    node.state: i for i, (_, _, node) in enumerate(frontier.heap)
                }

                if next_state not in state_indices:
                    frontier.push(next_node, next_priority)
                else:
                    existing_index = state_indices[next_state]
                    existing_priority = frontier.heap[existing_index][2].cost

                    if next_priority < existing_priority:
                        frontier.update(next_node, next_priority)

                # if next_state in frontier.entry_finder:
                #     return
                # frontier.update(next_node, next_priority)


# Weighted A* Search
# def a_star(initial_state, successor, goal_test, heuristic):
#     initial_node = Node(initial_state, [], 0)
#     frontier = [(initial_node, heuristic(initial_state))]
#     explored = []

#     while frontier:
#         curr_node, _ = heapq.heappop(frontier)
#         print(
#             f"Exploring node: {curr_node.state} with cost: {curr_node.cost} and heuristic: {heuristic(curr_node.state)}"
#         )
#         explored.append(curr_node.state)

#         if goal_test(curr_node.state):
#             print("Goal!")
#             print(f"Current state: {curr_node.state}")
#             print(f"Current path:\n")
#             pprint(curr_node.path)
#             return curr_node.path

#         for action, next_state, step_cost in successor(curr_node.state):
#             next_cost = curr_node.cost + step_cost
#             next_node = Node(next_state, curr_node.path + [action], next_cost)
#             next_priority = next_cost + heuristic(next_state)  # * W

#             if next_state not in explored and all(
#                 next_state != node.state for node, _ in frontier
#             ):
#                 heapq.heappush(frontier, (next_node, next_priority))
#             else:
#                 for i, (node, priority) in enumerate(frontier):
#                     if node.state == next_state and priority > next_priority:
#                         frontier[i] = (next_node, next_priority)
#                         heapq.heapify(frontier)
#                         break


def plot_path(path, start_str, dest_str):
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
                tooltip=f"Step {i + 1}: Waypoint\nDrive for {round(action[1])} minutes",
                icon=folium.Icon(color="black"),
            ).add_to(route_map)
        elif action[0] == "Station":
            folium.Marker(
                location=action[2],
                popup=action[2],
                tooltip=f"Step {i + 1}: Station\nCharge for {round(action[1])} minutes",
                icon=folium.Icon(color="red"),
            ).add_to(route_map)

    route_map.save(f"routes/{start_str}_{dest_str}_{Constants.MAX_RANGE}.html")


def plot_google_maps_route(path):
    start = path[0][2]
    end = path[-1][2]
    waypoints = [(f"{lat},{lng}") for _, _, (lat, lng) in path[1:-1]]
    waypoints_str = "|".join(waypoints)

    # Build the URL
    directions_url = f"https://maps.googleapis.com/maps/api/directions/json?origin={start[0]},{start[1]}&destination={end[0]},{end[1]}&waypoints={waypoints_str}&key={google_api_key}&mode=driving"

    # Make the request
    response = requests.get(directions_url)
    directions_data = json.loads(response.text)

    # Get the total driving time in minutes
    total_driving_time = 0
    for leg in directions_data["routes"][0]["legs"]:
        total_driving_time += leg["duration"]["value"] / 60  # convert to minutes

    # Get the total charging time
    total_charging_time = 0
    for _, action in enumerate(path):
        if action[0] == "Station":
            total_charging_time += action[1]

    total_time = total_driving_time + total_charging_time

    print(f"Total driving time: {round(total_driving_time)} minutes")
    print(f"Total charging time: {round(total_charging_time)} minutes")
    print(f"Total time: {round(total_time)} minutes")

    # Build the URL for opening in a web browser
    browser_url = f"https://www.google.com/maps/dir/?api=1&origin={start[0]},{start[1]}&destination={end[0]},{end[1]}&waypoints={waypoints_str}&key={google_api_key}&travelmode=driving"
    webbrowser.open(browser_url)


def plot_all_nodes(start_str, dest_str):
    path = [("Waypoint", 0, waypoint) for waypoint in Constants.WAYPOINTS] + [
        ("Station", 0, station) for station in Constants.STATIONS
    ]
    plot_path(path, start_str, dest_str)


def main():
    DEFAULT = dict(
        max_range=300,
        initial_range=300,
        time_to_charge=25,
        station_dist=5,
        goal_dist=0.2,
        algorithm="a_star",
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
        "-a",
        "--algorithm",
        help="Algorithm to be used to find a route.",
        required=False,
        default=DEFAULT["algorithm"],
        choices=["a_star", "ucs", "bfs", "dfs", "show_nodes"],
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
        "--time-to-charge",
        help="Time (min) to charge from 10% to 80% (DC Fast).",
        required=False,
        default=DEFAULT["time_to_charge"],
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

    # Initialize simple constants from command line arguments
    Constants.MAX_RANGE = args.max_range
    Constants.MIN_THRESHOLD = 0.1 * Constants.MAX_RANGE
    Constants.MAX_THRESHOLD = 0.8 * Constants.MAX_RANGE
    Constants.TIME_TO_CHARGE = args.time_to_charge
    Constants.DIST_FROM_STATION = args.station_dist
    Constants.DIST_FROM_GOAL = args.goal_dist

    # Initialize CHARGE_RATE from TIME_TO_CHARGE
    Constants.CHARGE_RATE = (
        Constants.MAX_THRESHOLD - Constants.MIN_THRESHOLD
    ) / Constants.TIME_TO_CHARGE

    # Initialize START (lat, lng) pair by geocoding
    start_str = args.start
    start_geocode_result = gmaps.geocode(start_str)[0]["geometry"]["location"]
    Constants.START = (
        start_geocode_result["lat"],
        start_geocode_result["lng"],
    )

    # Initialize DESTINATION (lat, lng) pair by geocoding
    dest_str = args.dest
    dest_geocode_result = gmaps.geocode(dest_str)[0]["geometry"]["location"]
    Constants.DESTINATION = (
        dest_geocode_result["lat"],
        dest_geocode_result["lng"],
    )

    # Initialize initial_range and initial_state
    initial_range = (
        args.initial_range
        if args.initial_range <= Constants.MAX_RANGE
        else Constants.MAX_RANGE
    )

    initial_state = (Constants.START, initial_range)

    # Initialize WAYPOINTS list, including start and destination locations
    Constants.WAYPOINTS = get_waypoints_along_route(
        Constants.START, Constants.DESTINATION
    )
    Constants.WAYPOINTS.insert(0, Constants.START)
    Constants.WAYPOINTS.append(Constants.DESTINATION)

    # Initialize STATIONS list
    Constants.STATIONS = get_stations_along_route(
        Constants.START, Constants.DESTINATION
    )

    # Initialize ROUTE_GRAPH
    Constants.ROUTE_GRAPH = generate_route_graph()

    # Initialize AVERAGE_SPEED
    directions_result = gmaps.directions(
        Constants.START, Constants.DESTINATION, mode="driving"
    )
    duration = directions_result[0]["legs"][0]["duration"]["value"] / 3600  # hours
    distance = directions_result[0]["legs"][0]["distance"]["value"] / 1609  # miles
    Constants.AVERAGE_SPEED = distance / duration

    if args.algorithm == "a_star":
        cProfile.runctx(
            "a_star(initial_state, successor, goal_test, heuristic)",
            globals=dict(
                a_star=a_star,
                initial_state=initial_state,
                successor=successor,
                goal_test=goal_test,
                heuristic=heuristic,
            ),
            locals={},
        )
        return
        # final_path = a_star(initial_state, successor, goal_test, heuristic)
    elif args.algorithm == "ucs":
        final_path = ucs(initial_state, successor, goal_test)
    elif args.algorithm == "bfs":
        final_path = bfs(initial_state, successor, goal_test)
    elif args.algorithm == "dfs":
        final_path = dfs(initial_state, successor, goal_test)
    elif args.algorithm == "show_nodes":
        plot_all_nodes(start_str, f"{dest_str}_all-nodes")
        return
    else:
        print(f"Must use valid algorithm.")
        return

    if final_path:
        plot_path(final_path, start_str, dest_str)
        # plot_google_maps_route(final_path)
    else:
        print(f"No route found!")


if __name__ == "__main__":
    main()
    # cProfile.run("main()")
