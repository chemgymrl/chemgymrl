import numpy as np
import math


class Shelf:
    def __init__(self,
                 # shelf_size,
                 shelf_tiers):
        # self.shelf_size = shelf_size
        self.shelf_tiers = shelf_tiers

        self.tiers = []
        for i in range(self.shelf_tiers):
            self.tiers.append({})

    def put_vessel(self,
                   vessel,
                   shelf,
                   tier,
                   ):
        shelf.tiers[tier][vessel.label] = vessel

    def get_vessel(self,
                   vessel_label,
                   shelf,
                   tier,
                   ):
        return shelf.tiers[tier].pop(vessel_label)
