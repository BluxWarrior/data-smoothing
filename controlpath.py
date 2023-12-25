import json
import re

def extract_coordinate(coord_string):
    match = re.search(r'POINT\(([^ ]+) ([^ ]+)\)', coord_string)
    if match:
        return float(match.group(1)), float(match.group(2))
    else:
        return None, None

def encode_coordinates(coordinates):
    def encode_value(value):
        value = int(round(value * 1e5))
        value <<= 1
        if value < 0:
            value = ~value
        chunks = []
        while value >= 0x20:
            chunks.append((0x20 | (value & 0x1f)) + 63)
            value >>= 5
        chunks.append(value + 63)
        return chunks

    # Separate longitudes and latitudes
    longitudes, latitudes = zip(*coordinates)

    coords = zip(latitudes, longitudes)

    encoded_string = ""
    last_lat, last_lng = 0, 0
    for lat, lng in coords:
        delta_lat = encode_value(lat - last_lat)
        delta_lng = encode_value(lng - last_lng)
        last_lat, last_lng = lat, lng
        encoded_string += ''.join(map(chr, delta_lat + delta_lng))
    return encoded_string

def extract_coordinates(file_path):
    with open(file_path, 'r') as file:
        data_json = json.loads(file.read())

    # Extract coordinates
    coordinates = [extract_coordinate(entry['coordinates']) for entry in data_json]
    return coordinates

def extract_path(coordinates, pathID):
    path = []
    for id in pathID:
        path.append(coordinates[id])
    return path

# coordinates = extract_path("./GPS_DATA/walks/walkitlikedog.json")
# print(coordinates)
# print(encode_coordinates(coordinates))