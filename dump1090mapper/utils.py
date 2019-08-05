from math import radians, cos, sin, asin, sqrt, pi, atan2, degrees
import geopandas as gpd
from functools import reduce
from io import BytesIO
import json
from datetime import timedelta

COPTERS = {
  "N917PD": "ACB1F5",
  "N918PD": "ACB5AC",
  "N919PD": "ACB963",
  "N920PD": "ACBF73",
  "N319PD": "A36989",
  "N422PD": "A50456",
  "N414PD": "A4E445",
  "N23FH" : "A206AC",
  "N509PD": "A65CA8",
}

MAX_MINUTES_BETWEEN_TRAJECTORIES = 10 # minutes
MAX_UNMARKED_INTERPOLATION_SECS = 60 # seconds; how long between points to connect with a solid versus dotted line

DEFAULT_CRS = 4326
def load_json_to_gdf(geojson_or_topojson_fn, crs=None):
    gdf = gpd.GeoDataFrame.from_file(geojson_or_topojson_fn)
    gdf.crs = {'init' :f"epsg:{DEFAULT_CRS}"}
    if crs != DEFAULT_CRS:
        gdf = gdf.to_crs(epsg=crs)
    return gdf

def split_rows_at_timediff(rows_most_recent_first, timediff=MAX_MINUTES_BETWEEN_TRAJECTORIES):
    try: 
        index_of_first_record_after_gap = next(i for (i, row) in enumerate(rows_most_recent_first[1:]) if  row["timediff"] > 60*timediff) + 1
        return rows_most_recent_first[:index_of_first_record_after_gap]
    except StopIteration:
        return rows_most_recent_first

# My original thought was points should be ordered by time from the aircraft, generated_datetime; and
# and whether to tweet should be based on parsed time)
# But generated_datetime and logged_datetime differ (from parsed_time) by 5 hours for squitters from the two sites. Odd!
#      
# I think we have to do something more complicated, otherwise you end up with stuff like this
# https://twitter.com/nypdhelicopters/status/1097569009455845376 due to generated_time being the right order
# and parsed_time being the wrong order (but guaranteed right hour). (node mapify.js ACB1F5 N917PD '2019-02-18 18:40:17' '2019-02-18 18:43:11')
#     
# An example of having multiple sites with mis-ordered generated_time is... TK
# 
# Moderately-successful solution is to calculate the difference in hours between the sites and 'correct' generated_datetime that way.
def _hour_differences(rows):
    # this finds a time per client at the time closest to when we have one time from each
    # so that they *should* be close enough in time to each other that we don't end up an hour off.
    # e.g. {0: 0, 1: -4} to show that client 1 has a relative time difference of -4 hours from client ID zero.
    unique_client_ids = set(row["client_id"] for row in rows)
    one_time_per_client = { client_id: next(row["generated_datetime"] for row in rows if row["client_id"] == client_id) for client_id in unique_client_ids }

    first_client_id, first_client_time = one_time_per_client.popitem()
    # if the difference is 50+ minutes, it's an hour difference. Less than that, we assume it's because we're comparing points that were legitimately seen at different times.
    hour_differences = {client_id: ( (time - first_client_time) + timedelta(minutes=10)).total_seconds() // 3600  for client_id, time in one_time_per_client.items()}
    hour_differences[first_client_id] = 0
    return hour_differences

def sort_rows_by_corrected_time(rows):
    hour_differences = _hour_differences(rows)
    print(hour_differences)
    for row in rows:
        row["corrected_time"] = row["generated_datetime"] - timedelta(hours=hour_differences[row["client_id"]])

    return sorted(rows, key=lambda x: x["corrected_time"])

def group_rows_between_gaps(rows_most_recent_last):
    """ create a list of grouped sub-trajectories (most recent group first, most recent row first in group)

    grouped either into those whose constituent points are 
     - separated by under MAX_UNMARKED_INTERPOLATION_SECS (e.g. 30 secs)
     - separated by more
    so that those separated by more can be marked with a dashed line to signal interpolation.
    """
    assert rows_most_recent_last[0]["corrected_time"] <= rows_most_recent_last[1]["corrected_time"]
    def _reducer(memo, row):
        if len(memo) == 0:
            return [[row]]
        if row["timediff"] > MAX_UNMARKED_INTERPOLATION_SECS:
            memo[-1].append(row) # the interpolated group (which is the last row of the previous group, plus the current row)
            memo.append([row])   # a new group
        else:
            memo[-1].append(row)
        return memo
    reduced = reduce(_reducer, rows_most_recent_last, [])
    if len(reduced[-1]) <= 1:
        reduced = reduced[:-1]
    return reduced

def rows_to_geojson(this_trajectory_rows_grouped):
    features = []
    for group_of_rows_most_recent_last in this_trajectory_rows_grouped:
        assert len(group_of_rows_most_recent_last) == 1 or group_of_rows_most_recent_last[0]["corrected_time"] <= group_of_rows_most_recent_last[1]["corrected_time"]
        features.append({
            "type": "Feature",
            "properties": {
              "interpolated": len(group_of_rows_most_recent_last) == 2 and group_of_rows_most_recent_last[1]["timediff"] > MAX_UNMARKED_INTERPOLATION_SECS,
              "traj_start": group_of_rows_most_recent_last[0]["corrected_time"].isoformat(),
              "traj_end": group_of_rows_most_recent_last[-1]["corrected_time"].isoformat()
            },
            "geometry": {
              "type": "LineString",
              "coordinates":  list([(row["lon"], row["lat"]) for row in group_of_rows_most_recent_last])
            }
        })
    linestring = {
        "type": "FeatureCollection",
    }
    linestring["features"] = features
    gdf = gpd.GeoDataFrame.from_file(BytesIO(json.dumps(linestring).encode()))
    gdf.crs = {'init' :f"epsg:{DEFAULT_CRS}"}
    return (gdf, linestring)

#### Geography methods
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 3956 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def point_at_distance(lon1_deg, lat1_deg, distance_mi, bearing_deg):
  R = 3958.8 # 3958.8 mi; 6371 km 
  bearing_rad = radians(bearing_deg)
  lat1 = lat1_deg * (pi / 180) # Current lat point converted to radians
  lon1 = lon1_deg * (pi / 180) # Current long point converted to radians
  lat2 = asin( sin(lat1)*cos(distance_mi/R) +
               cos(lat1)*sin(distance_mi/R)*cos(bearing_rad))
  lon2 = lon1 + atan2(sin(bearing_rad)*sin(distance_mi/R)*cos(lat1),
                       cos(distance_mi/R)-sin(lat1)*sin(lat2))
  return {"lat": lat2 / (pi / 180), "lon": lon2 / (pi / 180)}


def bearing(lon1, lat1, lon2, lat2):
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    diffLong = radians(lon2 - lon1)
    x = sin(diffLong) * cos(lat2)
    y = cos(lat1) * sin(lat2) - (sin(lat1)
            * cos(lat2) * cos(diffLong))
    initial_bearing = atan2(x, y)
    # Now we have the initial bearing but atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing