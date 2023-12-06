import heapq
import googlemaps
import requests

# Constants
MAX_RANGE = 405  # km
THRESHOLD = 0  # km
DIST_FROM_STATION = 15  # km
NEW_RANGE = MAX_RANGE
W = 3
START = "baldwinsville, ny"
DESTINATION = "syracuse, ny"
initial_state = (START, MAX_RANGE)

# Initialize Google Maps API client
google_api_file = open("google_api_key.txt", "r")
api_key = google_api_file.read()
google_api_file.close()
gmaps = googlemaps.Client(key=api_key)

# Initialize NREL API key
nrel_api_file = open("nrel_api_key.txt", "r")
nrel_api_key = nrel_api_file.read()
nrel_api_file.close()


class Node:
    def __init__(self, state, path, cost):
        self.state = state
        self.path = path
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost


# Return list of waypoints between two locations
def get_adjacent_waypoints(current_location, destination):
    # Request directions
    directions_result = gmaps.directions(current_location, destination, mode="driving")

    # Extract waypoints
    steps = directions_result[0]["legs"][0]["steps"]
    waypoints = [
        (step["end_location"]["lat"], step["end_location"]["lng"]) for step in steps
    ]

    return waypoints


# Return distance in km between two locations
def get_distance(start, end):
    # Request directions
    directions_result = gmaps.directions(start, end, mode="driving")

    # Extract distance in km
    distance = directions_result[0]["legs"][0]["distance"]["value"] / 1000

    return distance


# Return list of charging stations within a given range of a route
def get_stations_nearby_route(start, end):
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
        "radius": DIST_FROM_STATION * 0.621371,  # Convert to mi
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
            (station["latitude"], station["longitude"], station["distance"])
            for station in data["fuel_stations"]
        ]
        return stations
    else:
        # Handle cases where 'fuel_stations' is not in the response
        return []


# Return the nearest station within a distance of a given location
def get_nearest_station(location):
    url = "https://developer.nrel.gov/api/alt-fuel-stations/v1/nearest.json"

    params = {
        "api_key": nrel_api_key,
        "location": location,
        "radius": DIST_FROM_STATION * 0.621371,  # Convert to mi
        "fuel_type": "ELEC",
        "ev_charging_level": "dc_fast",
        "limit": 1,
        "status": "E",
        "access": "public",
    }

    # Make API request
    response = requests.get(url, params=params)
    data = response.json()

    # Check if 'fuel_stations' key exists in the response
    if "fuel_stations" in data and data["fuel_stations"]:
        stations = [
            (station["latitude"], station["longitude"], station["distance"])
            for station in data["fuel_stations"]
        ]
        return stations
    else:
        # Handle cases where 'fuel_stations' is not in the response
        return []


# Return a list of successors in the form (action, state, step_cost)
def successor(state):
    current_location, current_range = state
    successors = []

    waypoints = get_adjacent_waypoints(current_location, DESTINATION)
    stations = get_stations_nearby_route(current_location, DESTINATION)

    for waypoint in waypoints:
        distance_to_waypoint = get_distance(current_location, waypoint)

        # Add waypoint as a successor if within range
        if current_range >= distance_to_waypoint:
            new_range = current_range - distance_to_waypoint
            next_state = (waypoint, new_range)
            step_cost = distance_to_waypoint
            successors.append(("Drive", next_state, step_cost))

    if stations:
        for station in stations:
            distance_to_station = get_distance(current_location, station)

            # Add charging station as a successor if within range
            if current_range >= distance_to_station + THRESHOLD:
                next_state = (station, NEW_RANGE)
                step_cost = distance_to_station
                successors.append(("Charge", next_state, step_cost))

    return successors


def goal_test(state):
    current_location, current_range = state
    return current_location == DESTINATION and current_range >= THRESHOLD


STATIONS = get_stations_nearby_route(START, DESTINATION)


def heuristic(state):
    current_location, current_range = state

    distance_to_destination = get_distance(current_location, DESTINATION)
    if current_range >= distance_to_destination:
        return distance_to_destination

    # # Find nearest charging station
    # nearest_station = get_nearest_station(current_location)[0]

    # return nearest_station[2]
    nearest_station = min(
        STATIONS, key=lambda station: get_distance(current_location, station)
    )
    distance_to_station = get_distance(current_location, nearest_station)
    return distance_to_station


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


path = a_star(
    initial_state,
    lambda state: successor(state),
    lambda state: goal_test(state),
    lambda state: heuristic(state),
)

print(path)

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
