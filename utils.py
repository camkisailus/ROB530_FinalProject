import yaml
from pgm_reader import Reader
import numpy as np
from map import Map


def load_maps():
    map_names=["bedroom","dining_room","garage","obstacle_world","study","turtle_world"]
    maps = {}
    for map_name in map_names:
        reader = Reader()
        image = reader.read_pgm("data/maps/"+map_name+".pgm")
        width = reader.width
        map = np.array(image)

        maps[map_name]= map
        with open("data/maps/"+map_name+".yaml", "r") as stream:
            try:
                mapdata = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
            resolution = mapdata['resolution']
            origin = mapdata['origin']
        map_obj = Map(map, origin, resolution)
        maps[map_name] = map_obj

    return maps

