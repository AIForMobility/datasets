from enum import Enum


class VehicleDetectionObjectNameId(Enum):
    E_SCOOTER = 0
    CAR = 1
    MOTORCYCLE = 2
    BICYCLE = 3
    SEGWAY = 4
    LICENCE_PLATE = 5


class HelmetObjectNameId(Enum):
    WITHOUT_HELMET = 0
    WITH_HELMET = 1
