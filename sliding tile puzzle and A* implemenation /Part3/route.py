#!/usr/local/bin/python3
# route.py : Find routes through maps
#
# Code by: Niveditha Bommanahally Parameshwarappa(nibomm), Dhruvil Mansukhbhai Dholariya(ddholari), Bindu Madhavi Dokala(bdokala)
#
# Based on skeleton code by B551 Course Staff, Fall 2023
#


# !/usr/bin/env python3
import sys
import heapq
import itertools
from math import tanh
from math import sqrt
import copy

# Helping function to read road-segments.txt file
def prepare_city_graph():
    # city_graph has entry for each city, corresponding value is list of tuple for each destination city
    # tuple of each destination city stores name of destination, distance, time, probability, and highway
    city_graph = dict()

    with open("road-segments.txt", 'r') as file:
        line = file.readline()
        while(line):
            src, dst, mile, speed_limit, highway = line[:-1].split(" ")
            mile = float(mile)
            speed_limit = float(speed_limit)
            if(src not in city_graph):
                city_graph[src] = list()
            if(dst not in city_graph):
                city_graph[dst] = list()
    
            normal_time = mile/speed_limit
            probabilty = 0
            if(speed_limit > 50):
                probabilty = tanh(mile/1000)
                
            city_graph[src].append((dst, mile, normal_time, probabilty, highway))
            city_graph[dst].append((src, mile, normal_time, probabilty, highway))
            
            line = file.readline()
    return city_graph

# Helping function to read city-gps.txt file
def prepare_city_lat_long():
    city_lat_long = dict()

    with open("city-gps.txt", 'r') as file:
            line = file.readline()
            while(line):
                city, lat, long_ = line[:-1].split(" ")
                city_lat_long[city] = (float(lat), float(long_))
                line = file.readline()
    return city_lat_long

# Heuristic cost function
def heuristic_cost(src, dst, cost_type, city_graph, city_lat_long):
    src_lat = 0.0
    src_long = 0.0
    dst_lat = 0.0
    dst_long = 0.0

    if(src in city_lat_long):
        src_lat, src_long = city_lat_long[src]
    else:
        min = 2**30
        for city, mile, _, _, _ in city_graph[src]:
            if(mile < min and city in city_lat_long):
                min = mile
                src_lat, src_long = city_lat_long[city]

    if(dst in city_lat_long):
        dst_lat, dst_long = city_lat_long[dst]
    else:
        min = 2**30
        for city, mile, _, _, _ in city_graph[dst]:
            if(mile < min and city in city_lat_long):
                min = mile
                dst_lat, dst_long = city_lat_long[city]

    if(src_lat == 0.0 or dst_lat == 0.0):
        return 0

    if(cost_type == "distance"):
        return sqrt((dst_lat-src_lat) ** 2 + (dst_long-src_long) ** 2) * 20.0
    elif(cost_type == "segments"):
        return sqrt((dst_lat-src_lat) ** 2 + (dst_long-src_long) ** 2) * 1.2
    else:
        return sqrt((dst_lat-src_lat) ** 2 + (dst_long-src_long) ** 2) * 0.01

def get_route(start, end, cost_type):
    city_graph = prepare_city_graph()
    city_lat_long = prepare_city_lat_long()

    counter = itertools.count()
    visited = set()
    node = {"segments" : 0, "distance" : 0, "time" : 0, "delivery" : 0, "route" : list()}
    fringe = list()
    heapq.heappush(fringe, (node[cost_type] + heuristic_cost(start, end, cost_type, city_graph, city_lat_long) , next(counter), start, node))

    while(fringe):
        cost, _, city, node = heapq.heappop(fringe)
        visited.add(city)
        
        if(city == end):
            break

        for dst, mile, time_, probability, highway in city_graph[city]:
            if(dst in visited):
                continue
                
            next_node = dict()
            next_node["segments"] = node["segments"] + 1
            next_node["distance"] = node["distance"] + mile
            next_node["time"] = node["time"] + time_
            next_node["delivery"] = node["delivery"] + time_ + probability * 2 * (node["delivery"] + time_)
            route = copy.deepcopy(node["route"])
            route.append((dst, "{} for {} miles".format(highway, mile)))
            next_node["route"] = route

            heapq.heappush(fringe, (next_node[cost_type] + heuristic_cost(dst, end, cost_type, city_graph, city_lat_long), 
                                    next(counter), dst, next_node))
    
    return {"total-segments" : node["segments"], 
            "total-miles" : node["distance"], 
            "total-hours" : node["time"], 
            "total-delivery-hours" : node["delivery"], 
            "route-taken" : node["route"]}


# Please don't modify anything below this line
#
if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise(Exception("Error: expected 3 arguments"))

    (_, start_city, end_city, cost_function) = sys.argv
    if cost_function not in ("segments", "distance", "time", "delivery"):
        raise(Exception("Error: invalid cost function"))

    result = get_route(start_city, end_city, cost_function)

    # Pretty print the route
    print("Start in %s" % start_city)
    for step in result["route-taken"]:
        print("   Then go to %s via %s" % step)

    print("\n          Total segments: %4d" % result["total-segments"])
    print("             Total miles: %8.3f" % result["total-miles"])
    print("             Total hours: %8.3f" % result["total-hours"])
    print("Total hours for delivery: %8.3f" % result["total-delivery-hours"])


