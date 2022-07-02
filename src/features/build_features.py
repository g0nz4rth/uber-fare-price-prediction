import holidays
import numpy as np
from math import sin, cos, asin, sqrt, radians


def check_holiday(date: str, country: str = 'US'):
    """
    This function checks if a given string date
    is a holiday on a given country.
    """
    country_holidays = holidays.country_holidays(country)
    
    return int(date in country_holidays)


def apply_heversine(latitude1, longitude1, latitude2, longitude2):
    """
    This function calculates the distance (in km) between 
    two coordinates (lat, long).
    """
    travel_dist = []
    
    for pos in range(len(longitude1)):
        long1,lati1,long2,lati2 = map(radians,[longitude1[pos],latitude1[pos],longitude2[pos],latitude2[pos]])
        dist_long = long2 - long1
        dist_lati = lati2 - lati1
        a = sin(dist_lati/2)**2 + cos(lati1) * cos(lati2) * sin(dist_long/2)**2
        c = 2 * asin(sqrt(a))*6371
        travel_dist.append(c)
       
    return travel_dist
