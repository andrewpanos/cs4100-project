# CS 4100: Artificial Intelligence:
### Andrew Panos & David Schaeffer

**Problem:**  
We are attempting to solve the problem of optimal route-finding, especially for longer trips, when driving an electric vehicle (EV), particularly in the United States. 
Most route-finding programs (e.g., directions in Apple or Google Maps) exclude refueling/recharging from their algorithm, operating under the assumption that the user 
will be able to refuel when necessary. Though this is usually a valid assumption when driving a gas vehicle due to the prevalence of gas stations throughout the United States, 
EV charging stations are less prevalent, especially in certain regions of the country. And when they are available, the existence of two different charging port standards 
precludes some EV owners from charging at certain stations. We aim to take guesswork and uncertainty out of route-finding by including in the route compatible EV charging stations. 
We also aim to minimize costs by finding routes that minimize the distance spent driving out of the way of the route to charge, the cost of charging, the time spent at charging 
stations, and the overall time and distance of the route. Each route will be specific to the vehicle driven: we will consider the compatibility of charging stations, estimated 
battery use and range, and preferred battery thresholds. This will reduce charging anxiety and allow EV drivers to confidently take long trips.
