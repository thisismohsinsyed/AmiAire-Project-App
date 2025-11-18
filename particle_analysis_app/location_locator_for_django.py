# location_locator_for_django.py
import googlemaps
from django.conf import settings

def get_lat_lng(location_name):
    """
    This function takes in a location name and uses the Google Maps Geocoding API 
    to return latitude and longitude as a dict. If geocoding fails, returns None.
    Example return:
    {
        'latitude': 12.345678,
        'longitude': 98.765432
    }
    """
    if not location_name:
        return None

    gmaps = googlemaps.Client(key=settings.GOOGLE_MAPS_API_KEY)
    geocode_result = gmaps.geocode(location_name)

    if geocode_result:
        lat_lng = geocode_result[0]['geometry']['location']
        return {
            'latitude': lat_lng['lat'],
            'longitude': lat_lng['lng']
        }
    else:
        return None
