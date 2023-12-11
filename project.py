import heapq
import googlemaps
import requests
import math
from pprint import pprint
from util import PriorityQueue
from geopy.distance import geodesic

# Initialize Google Maps API client
google_api_file = open("google_api_key.txt", "r")
api_key = google_api_file.read()
google_api_file.close()
gmaps = googlemaps.Client(key=api_key)

# Initialize NREL API key
nrel_api_file = open("nrel_api_key.txt", "r")
nrel_api_key = nrel_api_file.read()
nrel_api_file.close()

# Constants
MAX_RANGE = 300  # mi
THRESHOLD = 50  # mi
DIST_FROM_STATION = 10  # mi
GOAL_RADIUS = 0.5 # mi

# Generate start location lat/long pair
start_result = gmaps.geocode("baldwinsville, ny")[0]["geometry"]["location"]
START = (start_result["lat"], start_result["lng"])

# Generate destination location lat/long pair
dest_result = gmaps.geocode("boston, ma")[0]["geometry"]["location"]
DESTINATION = (dest_result["lat"], dest_result["lng"])

initial_state = (START, MAX_RANGE)

class Node:
    def __init__(self, state, path, cost):
        self.state = state
        self.path = path
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost


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

WAYPOINTS = get_waypoints_along_route(START, DESTINATION)


# # Return distance in miles between two locations
# def get_distance(start, end):
#     # Request directions
#     directions_result = gmaps.directions(start, end, mode="driving")

#     # Extract distance in miles
#     distance = directions_result[0]["legs"][0]["distance"]["value"] / 1609

#     return distance

# Return distance in miles between two locations
def get_distance(start, end):
    # Calculate the distance
    distance = geodesic(start, end).miles
    return distance

# Return list of charging stations within a given range of a route
def get_stations_along_route(start, end):
    url = "https://developer.nrel.gov/api/alt-fuel-stations/v1/nearby-route.json"

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

    params = {
        "api_key": nrel_api_key,
        "route": linestring,
        "radius": DIST_FROM_STATION,
        "fuel_type": "ELEC",
        "ev_charging_level": "dc_fast",
        "status": "E",
        "access": "public",
    }

    # Make API request
    response = requests.get(url, params=params)
    data = response.json()

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
        return []

STATIONS = get_stations_along_route(START, DESTINATION)

# # Return the nearest station within a distance of a given location
# def get_nearest_station(location):
#     url = "https://developer.nrel.gov/api/alt-fuel-stations/v1/nearest.json"

#     params = {
#         "api_key": nrel_api_key,
#         "radius": "infinite",  # Convert to mi
#         "fuel_type": "ELEC",
#         "ev_charging_level": "dc_fast",
#         "limit": 1,
#         "status": "E",
#         "access": "public",
#     }

#     if type(location) == tuple:
#         params.update({
#             "latitude": location[0],
#             "longitude": location[1]
#         })
#     elif type(location) == str:
#         params.update({
#             "location": location
#         })
#     else:
#         return []

#     # Make API request
#     response = requests.get(url, params=params)
#     data = response.json()

#     # Check if 'fuel_stations' key exists in the response
#     if "fuel_stations" in data and data["fuel_stations"]:
#         stations = [
#             (
#                 station["latitude"],
#                 station["longitude"],
#                 station["distance"],
#             )
#             for station in data["fuel_stations"]
#         ]

#         return stations[0]
#     else:
#         # Handle cases where 'fuel_stations' is not in the response
#         return []


# TODO: Convert this to edge cost
def step_cost(state, action):
    current_location, current_range = state
    action_str, distance = action

    if action_str == "Continue":
        return 1 / (1 + math.exp((current_range - distance) / THRESHOLD))
    elif action_str == "Divert":
        return 1 / (1 + math.exp(-(current_range - distance) / THRESHOLD))
    else:
        return

    # distance_to_destination = get_distance(current_location, DESTINATION)

    # Modify for threshold (SOC) at end of journey
    # if current_range >= distance_to_destination:
    #     return distance_to_destination

    # Find nearest charging station
    # nearest_station = get_nearest_station(current_location)[0]

    # return nearest_station[2]
    # nearest_station = min(
    #     STATIONS, key=lambda station: get_distance(current_location, station)
    # )
    # distance_to_station = get_distance(current_location, nearest_station)
    # distance_to_nearest_station = nearest_station[2]


# Return a list of successors in the form (action, state, step_cost)
def successor(state):
    current_location, current_range = state
    successors = []
    
    # filter waypoints and stations within range
    waypoints_within_range = filter(lambda waypoint: current_range - THRESHOLD >= get_distance(current_location, waypoint), WAYPOINTS)
    # print(list(waypoints_within_range))
    stations_within_range = filter(lambda station: current_range - THRESHOLD >= get_distance(current_location, station), STATIONS)
    # print(list(stations_within_range))

    # if status == "driving":
    #     # waypoints = get_waypoints_along_route(current_location, DESTINATION)

    #     lat, lng, distance = next_waypoint
    #     next_action = ("Continue", distance)
    #     if (current_range - THRESHOLD > distance):
    #         successors.append((next_action, ((lat, lng), current_range - distance, "driving"), step_cost(state, next_action)))

    #     # stations = get_stations_along_route(current_location, DESTINATION)
    #     lat, lng, distance = get_nearest_station(current_location)
    #     next_action = ("Divert", distance)
    #     successors.append((next_action, ((lat, lng), MAX_RANGE, "charging"), step_cost(state, next_action)))

    # if status == "charging":
    #     lat, lng, distance = get_next_waypoint(current_location, DESTINATION)
    #     next_action = ("Continue", distance)
    #     successors.append((next_action, ((lat, lng), current_range - distance, "driving"), step_cost(state, next_action)))

    for waypoint in waypoints_within_range:
        distance_to_waypoint = get_distance(current_location, waypoint)

        new_range = current_range - distance_to_waypoint
        next_state = (waypoint, new_range)
        next_action = ("Continue", distance_to_waypoint)
        successors.append((next_action, next_state, step_cost(state, next_action)))

    for station in stations_within_range:
        distance_to_station = get_distance(current_location, station)
        
        next_state = (station, MAX_RANGE)
        next_action = ("Divert", distance_to_station)
        successors.append(("Charge", next_state, step_cost(state, next_action)))

    return successors


def goal_test(state):
    current_location, current_range = state
    # current_location is within a certain distance of the destination
    distance_to_destination = get_distance(current_location, DESTINATION)
    return distance_to_destination <= GOAL_RADIUS and current_range >= THRESHOLD


# STATIONS = get_stations_nearby_route(START, DESTINATION)


def heuristic(state):
    current_location, current_range = state

    distance_to_destination = get_distance(current_location, DESTINATION)

    # Modify for threshold (SOC) at end of journey
    if current_range >= distance_to_destination:
        return distance_to_destination

    # Find nearest charging station
    nearest_station = get_nearest_station(current_location)[0]

    # return nearest_station[2]
    # nearest_station = min(
    #     STATIONS, key=lambda station: get_distance(current_location, station)
    # )
    # distance_to_station = get_distance(current_location, nearest_station)
    distance_to_nearest_station = nearest_station[2]

    return (
        distance_to_destination
        + math.exp(MAX_RANGE / current_range) * distance_to_nearest_station
    )

# Uniform Cost Search
def ucs(initial_state, successor, goal_test):
    initial_node = Node(initial_state, [], 0)
    frontier = PriorityQueue()
    frontier.push(initial_node, 0)
    explored = set()
    
    while not frontier.isEmpty():
        curr_node: Node = frontier.pop()
        curr_state = curr_node.state
        
        print(curr_state)

        explored.add(curr_state)

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
                    if item.state == next_state: # maybe not item.state
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
        print(f"Exploring node: {curr_node.state} with cost: {curr_node.cost}")
        explored.append(curr_node.state)

        if goal_test(curr_node.state):
            print("Goal state reached.")
            return curr_node.path

        for action, next_state, step_cost in successor(curr_node.state):
            next_cost = curr_node.cost + step_cost
            next_node = Node(next_state, curr_node.path + [action], next_cost)
            print(
                f"Adding successor: {next_state} with action: {action} and step cost: {step_cost}"
            )
            next_priority = next_cost + heuristic(next_state) * W

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


# path = a_star(
#     initial_state,
#     lambda state: successor(state),
#     lambda state: goal_test(state),
#     lambda state: heuristic(state),
# )

# print(path)

if __name__ == "__main__":
    ucs(initial_state, successor, goal_test)
    # print(get_nearest_station())

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
