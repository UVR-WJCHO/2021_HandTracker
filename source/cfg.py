import yaml

class Struct:

    def __init__(self, **params):
        self.__dict__.update(params)

with open ('C:/Research/2021_HandTracker/cfg/cfg.yaml', 'r') as f:
    params = yaml.safe_load(f)

parameters = Struct(**params)