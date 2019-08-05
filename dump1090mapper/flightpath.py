import matplotlib.pyplot as plt
import matplotlib as mpl
import geopandas as gpd
import pandas as pd
# import pysal as ps
# from pysal.contrib.viz import mapping as maps
from os.path import dirname, basename, abspath, join
import json
import pymysql.cursors
from io import BytesIO
from os import environ, getcwd
from datetime import datetime
from sys import stderr
from pyproj import Proj, transform
import fiona
from fiona.crs import from_epsg
from collections import Counter
from shapely.ops import linemerge
from shapely.geometry import MultiLineString
env = lambda x: environ.get(x, None)
from .utils import *



# IMAGE:
# TODO: trim edges to get rid of bled-over text
# TODO: state borders (ehh,  maybe not?)
# TODO: fiddle with text offsets/backgrounds
# TODO: factor out the plt.plot() stuff for the various lines that make up the plane
# TODO: double check the arrowheads being positioned right 

# DATA LOADING
# TODO: undo the reversing of records so much (i.e. in geojson and in plane-line plotting, sort_rows_by_corrected_time)
#       assert rows sort-order in the appropriate places
# TODO: ensure rows by corrected time (sort_rows_by_corrected_time) works
# TODO: from_json, FlightpathShingle.__init__() should actually call the parent init (DRY that out)
# TODO: should cache neighborhood names
# TODO: getting points from sql should have a time limit of like a week or a month
# TODO: run nypdcopterbot.rb with a saved JSON to test arrows, etc.

MIN_POINTS = 5
# TODO: MAX_MINUTES_BETWEEN_TRAJECTORIES = args.MAXTIMEDIFF or 10 # minutes
MAX_MINUTES_BETWEEN_TRAJECTORIES = 10 # minutes
SHINGLE_DURATION = 5 # minutes
SHINGLE_MIN_DURATION = 60 # seconds
MAP_MIN_POINTS = 10 # minimum points in a trajectory.
SHINGLE_DURATION_SECS = SHINGLE_DURATION * 60
class HelicopterShinglingError(Exception):
  pass
class HelicopterMappingError(Exception):
  pass



NYC_NEIGHBORHOOD_NAME_TRANSFORMS = {
    "park-cemetery-etc": None,
    "park-cemetery-etc-Brooklyn": None,
    "park-cemetery-etc-Bronx": None,
    "park-cemetery-etc-Manhattan": None,
    "park-cemetery-etc-Queens": None,
    "park-cemetery-etc-Staten Island": None,
    "North Side-South Side":  "Williamsburg"
}


needs_to_use_mysql = __name__ == "__main__" # TODO set this from parsed CLI args

global mysql_connection
mysql_connection = None


class Flightpath():
    def __init__(self, icao_hex, nnum=None, start_time=None, end_time=None, crs=DEFAULT_CRS):
        self.icao_hex = icao_hex
        self.nnum = nnum
        self.start_time = start_time
        self.end_time = end_time
        self.crs = crs or DEFAULT_CRS
        if (not not self.start_time) != (not not self.end_time):
            raise Exception("start_time and end_time must either both be specified or neither") 
        if self.start_time and self.start_time > self.end_time:
            raise Exception("Flightpath start_time must be before (less than) end_time")

        if self.start_time and self.start_time == self.end_time:
            raise Exception("Flightpath start_time must be STRICTLY before (less than) end_time; they were equal, which is weird.")


        self.plane_gdf = None
        self.plane_geojson = None
        self.plane_points_gdf = None # TODO factor this out, I think.
        self.trajectory_points = None 

        self.points_cnt = None
        self.groups_cnt = None

        self.neighborhoods_counter = NeighborhoodsCounter([
            (join(dirname(abspath(__file__)), "basemap/json/nynta_17a.json"), "NTAName"),
            (join(dirname(abspath(__file__)), "basemap/json/bridges_buffered.geojson"), "fullname"),
            ], 
            NYC_NEIGHBORHOOD_NAME_TRANSFORMS, 
            crs=self.crs)


        self.custom_map_fn = None

    def _connect_to_mysql(self):
        global mysql_connection
        self.mysql_connection = mysql_connection or pymysql.connect(host=env('MYSQLHOST'),
                                 user=env('MYSQLUSER') or env('MYSQLUSERNAME'),
                                 port=env('MYSQLPORT'),
                                 password=env('MYSQLPASSWORD'),
                                 db=env('MYSQLDATABASE') or "dump1090",
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)
        mysql_connection = self.mysql_connection

    def _default_map_fn(self):
        start_time_str = self.start_time.isoformat().replace(":", "_") # TODO factor this out in favor of a .format()
        end_time_str = self.end_time.isoformat().replace(":", "_")
        return f"{self.nnum or self.icao_hex}-{start_time_str}-{end_time_str}.png"

    def set_map_fn(self, fn):
        self.custom_map_fn = fn

    def get_map_fn(self):
        return self.custom_map_fn or self._default_map_fn()

    @classmethod
    def from_json(cls, fn):
        """creates a Flightpath from JSON"""
        with open(fn, 'r') as f:
            flightpath_json = json.loads(f.read())
        fp = Flightpath(flightpath_json["icao_hex"], flightpath_json["nnum"], datetime.fromisoformat(flightpath_json["start_time"]), datetime.fromisoformat(flightpath_json["end_time"]), crs=flightpath_json["crs"])
        fp.plane_geojson = flightpath_json["geojson"]
        fp.plane_gdf = gpd.GeoDataFrame.from_file(BytesIO(json.dumps(flightpath_json["geojson"]).encode()))
        fp.plane_gdf.crs = {'init' :f"epsg:4236"} # it's saved in latlong!

        trajectory_points = flightpath_json["trajectory_points"]
        for point in trajectory_points:
            point["corrected_time"] = datetime.fromisoformat(point["corrected_time"])
        fp.trajectory_points = trajectory_points
        if fp.crs != DEFAULT_CRS:
            fp.plane_gdf = fp.plane_gdf.to_crs(epsg=fp.crs)
        fp.points_cnt = len(fp.trajectory_points)
        fp.groups_cnt = len(fp.plane_geojson["features"])
        return fp

    def centerpoint(self):
        centerpoint = linemerge(MultiLineString(list(plane["geometry"]))).centroid.coords[0]
        centerpoint_lat_lon = transform(Proj(init=f"EPSG:{crs}") , Proj(init='EPSG:4326'), *centerpoint)
        return {"x": centerpoint[0], "y": centerpoint[1], "lat": centerpoint_lat_lon[1], "lon": centerpoint_lat_lon[0]}

    def neighborhoods_underneath(self):
        neighborhood_names = self.neighborhoods_counter.neighborhoods_under_lines(self.plane_gdf)
        trajectory_points_df = pd.DataFrame(self.trajectory_points)
        trajectory_points_gdf = gpd.GeoDataFrame(trajectory_points_df, geometry=gpd.points_from_xy(trajectory_points_df.lat, trajectory_points_df.lon), crs=4326)
        if self.crs != DEFAULT_CRS:
            trajectory_points_gdf = trajectory_points_gdf.to_crs(epsg=self.crs)
        neighborhood_names = self.neighborhoods_counter.neighborhoods_under_points(trajectory_points_gdf)
        return neighborhood_names

    def json_fn(self):
        start_time_str = self.start_time.isoformat().replace(":", "_")
        end_time_str = self.end_time.isoformat().replace(":", "_")
        return f"{self.nnum or self.icao_hex}-{start_time_str}-{end_time_str}.flightpath.json"


    def to_json(self):
        """saves a JSON file suitable to be re-loaded with from_json"""
        if self.plane_gdf is None:
            self._get_points_from_sql()
        trajectory_points = []
        for point in self.trajectory_points: # make a copy of trajectory_points with datetz objects that are serializable
            point = dict(point)
            point["corrected_time"] = point["corrected_time"].isoformat()
            trajectory_points.append(point)
        return json.dumps({
            "geojson": self.plane_geojson,
            "icao_hex": self.icao_hex,
            "nnum": self.nnum,
            "start_time": self.start_time.isoformat(),
            "end_time":  self.end_time.isoformat(),
            "crs": self.crs,
            "trajectory_points": trajectory_points
        }, sort_keys=True,
           indent=4, 
           separators=(',', ': '))

    def _get_points_from_sql(self):
        self._connect_to_mysql()
        print("getting points from SQL")
        with mysql_connection.cursor() as cursor:
            sql = f"""
                SELECT *, 
                  convert_tz(parsed_time, '+00:00', 'US/Eastern') datetz, 
                  conv(icao_addr, 10,16) as icao_hex
                FROM squitters 
                WHERE icao_addr = conv('{self.icao_hex}', 16,10) and 
                  lat is not null
                TIMECONDITIONGOESHERE
                order by parsed_time desc;""".replace(r"\s+", " ").strip()
            if self.start_time:
                start_time_str = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
                end_time_str   = self.end_time.strftime("%Y-%m-%d %H:%M:%S")
                sql = sql.replace("TIMECONDITIONGOESHERE", f"AND parsed_time >= '{start_time_str}' AND parsed_time <= '{end_time_str}'")
            else:
                sql = sql.replace("TIMECONDITIONGOESHERE", '')
            cursor.execute(sql)
            rows = cursor.fetchall() # most recent first.
            lats_count = 0
            for row in rows:
                if row["lat"]:
                    lats_count += 1
                    row["lat"] = float(row["lat"]) # lat/lons come out of pymysql as Decimal which sucks and isn't JSON serializable
                    row["lon"] = float(row["lon"]) # lat/lons come out of pymysql as Decimal which sucks and isn't JSON serializable
            if lats_count < MIN_POINTS:
              return None, None, None

        for row1, row2 in zip(rows[:-1], rows[1:]):
            row2["timediff"] = abs((row1["datetz"] - row2["datetz"]).total_seconds())

        if not self.start_time:
            trajectory_points = sort_rows_by_corrected_time(split_rows_at_timediff(rows, MAX_MINUTES_BETWEEN_TRAJECTORIES))
        else:
            trajectory_points = sort_rows_by_corrected_time(rows)

        for row1, row2 in zip(trajectory_points[:-1], trajectory_points[1:]):
            row2["timediff"] = abs((row1["datetz"] - row2["datetz"]).total_seconds())

        # sorted with the most recent point last.
        assert trajectory_points[0]["corrected_time"] <= trajectory_points[1]["corrected_time"] and trajectory_points[0]["corrected_time"] <= trajectory_points[-1]["corrected_time"]

        self.trajectory_points = [{"corrected_time": row["corrected_time"], "timediff": row.get("timediff", 0), "lat": row["lat"], "lon": row["lon"]} for row in trajectory_points]
        trajectory_points_grouped = group_rows_between_gaps(self.trajectory_points)
        for group in trajectory_points_grouped:
            assert(len(group) >= 2)
        plane, plane_geojson = rows_to_geojson(trajectory_points_grouped)
        self.points_cnt = len(self.trajectory_points)
        print(f"got {self.points_cnt} points from SQL for this trajectory")
        self.groups_cnt = len(trajectory_points_grouped)
        if self.crs != DEFAULT_CRS:
            plane = plane.to_crs(epsg=self.crs)
        self.plane_gdf = plane
        self.plane_geojson = plane_geojson
        self.start_time = self.trajectory_points[0]["corrected_time"]
        self.end_time =   self.trajectory_points[-1]["corrected_time"]

    def as_shingles(self):
        """returns a list of other Flightpaths"""
        if self.plane_gdf is None:
            self._get_points_from_sql()
        # plane_points = [item for sublist in [list(n.coords) for n in list(self.plane_gdf["geometry"])] for item in sublist]

        traj_duration_secs = abs((self.trajectory_points[-1]['corrected_time'] - self.trajectory_points[0]['corrected_time']).total_seconds()) 
        shingles_cnt = int((traj_duration_secs / (SHINGLE_DURATION_SECS)) * 2) # how many shingles to generate
        print(f"shingles count: {shingles_cnt} over {int(traj_duration_secs/60)} min")
        for i in range(shingles_cnt):
            shingle_start_elapsed_time = ((SHINGLE_DURATION_SECS/2.0) * i)
            shingle_end_elapsed_time  = ((SHINGLE_DURATION_SECS/2.0) * (i + 2))
            print(shingle_start_elapsed_time)
            shingle_start_time = self.trajectory_points[0]['corrected_time'] + timedelta(seconds=shingle_start_elapsed_time) # self.trajectory_points[-1] is the one that happened first.
            shingle_end_time = self.trajectory_points[0]['corrected_time'] + timedelta(seconds=shingle_end_elapsed_time) # self.trajectory_points[-1] is the one that happened first.
            shingle_points = [pt for pt in self.trajectory_points if pt['corrected_time'] >= shingle_start_time and pt['corrected_time'] <= shingle_end_time]
            print(f"Shingle has {len(shingle_points)} points")

            if len(shingle_points) >= MAP_MIN_POINTS and (shingle_points[-1]['corrected_time'] - shingle_points[0]['corrected_time']).total_seconds() >= SHINGLE_MIN_DURATION:
                print(shingle_start_time, shingle_end_time)
                yield FlightpathShingle(self, shingle_points)


    def to_map(self, fn=None, arbitrary_marker=None, include_background=True, background_color=None):
        if self.plane_gdf is None:
            self._get_points_from_sql()

        if self.points_cnt < MAP_MIN_POINTS:
            raise HelicopterMappingError("too few points")
            return 
        fig = plt.figure(frameon=False)
        ax = plt.axes()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        if background_color:
            ax.set_facecolor(background_color)


        ## figuring out the right bounds for the map, which we want to be a square.
        # The latitude is 40.730610, and the longitude is -73.935242
        this_trajectory_lats = [i for sl in [ [coord[1] for coord in feat["geometry"]["coordinates"]] for feat in  self.plane_geojson["features"]] for i in sl]
        this_trajectory_lons = [i for sl in [ [coord[0] for coord in feat["geometry"]["coordinates"]] for feat in  self.plane_geojson["features"]] for i in sl]
        max_lat = max(this_trajectory_lats)
        min_lat = min(this_trajectory_lats)
        avg_lat = sum(this_trajectory_lats) / len(this_trajectory_lats)
        mid_lat = (max_lat + min_lat) / 2
        lat_range = max_lat - min_lat
        max_lon = max(this_trajectory_lons)
        min_lon = min(this_trajectory_lons)
        avg_lon = sum(this_trajectory_lons) / len(this_trajectory_lons)
        mid_lon = (max_lon + min_lon) / 2
        lon_range = max_lon - min_lon
        ylength = haversine(min_lon, min_lat, min_lon, max_lat) # mi
        xlength = haversine(min_lon, min_lat, max_lon, min_lat) # mi

        window_length = max(xlength, ylength)
        lat_min = point_at_distance(mid_lon, mid_lat, window_length/1.9, 180 )["lat"]
        lat_max = point_at_distance(mid_lon, mid_lat, window_length/1.9, 0 )["lat"]
        lon_min = point_at_distance(mid_lon, mid_lat, window_length/1.9, 270 )["lon"]
        lon_max = point_at_distance(mid_lon, mid_lat, window_length/1.9, 90 )["lon"]
        orig_proj = Proj(init='EPSG:4326') 
        dest_proj = Proj(init=f"EPSG:{self.crs}") 

        xmin,ymin = transform(orig_proj, dest_proj, lon_min, lat_min)
        xmax,ymax = transform(orig_proj, dest_proj, lon_max, lat_max)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.update_datalim(((xmin, ymin), (xmax, ymax)))
        ax.set_aspect('equal', 'box')

        counties_topojson_fn = "basemap/json/counties.json"
        bridges_topojson_fn = "basemap/json/bridges.json"
        parks_airports_topojson = "basemap/json/nyc_parks_airports.json"
        runways_topojson = "basemap/json/airports.json"

        if include_background:
            # counties
            # ny_counties = ny_counties[ny_counties['BoroName'].notna()] + ny_counties[ny_counties['STATEFP'] == '09']  + ny_counties[ny_counties['STATEFP'] == '34']
            nynjct_counties = load_json_to_gdf(join(pwd, counties_topojson_fn), self.crs)
            nynjct_counties.plot(ax=ax, color='#ffffca', edgecolor='black')

            # TODO: draw state boundaries and county boundaries (by zipping all the counties together and drawing the border differently based on whether the state is the same for both counties)
            # see http://darribas.org/gds15/content/labs/lab_03.html

            bridges = load_json_to_gdf(join(pwd, bridges_topojson_fn), self.crs)
            bridges.plot(ax=ax, color='#dddddd', edgecolor='#dddddd', linewidth=0.5)


            # Parks, airports
            parks_airports = load_json_to_gdf(join(pwd, parks_airports_topojson), self.crs)
            parks = parks_airports[parks_airports["ntaname"] != "Airport"]
            airports_shapes = parks_airports[parks_airports["ntaname"] == "Airport"]
            airports_shapes.plot(ax=ax, color="#ffcccc")
            parks.plot(ax=ax, color="#339933")

            runways = load_json_to_gdf(join(pwd, runways_topojson), self.crs)
            runways.plot(ax=ax, edgecolor="#ffffff")

        if arbitrary_marker:
            plt.plot(*arbitrary_marker, 'bo', markersize=8) 


        ### drawing the plane
        # with pointy arrows at the end
        if self.plane_gdf["geometry"].shape[0] == 1: 
            #  if there's only one line, just draw it (but break it up into two pieces so we can put arrows on the ends.)
            plane_only_line_xy = self.plane_gdf["geometry"][0].xy
            plane_first_line_xy = (plane_only_line_xy[0][:2], plane_only_line_xy[1][:2])
            plane_last_line_xy = (plane_only_line_xy[0][1:], plane_only_line_xy[1][1:])
            plane_last_line_props = plane_first_line_props = self.plane_geojson["features"][0]
        else: # 2 + segments
            plane_first_line = self.plane_gdf["geometry"][0]
            plane_last_line = self.plane_gdf["geometry"][len(self.plane_gdf["geometry"])-1]
            plane_first_line_xy = plane_first_line.xy
            plane_last_line_xy = plane_last_line.xy
            plane_first_line_props = self.plane_geojson["features"][0]
            plane_last_line_props  = self.plane_geojson["features"][-1]

        if self.plane_gdf["geometry"].shape[0] >= 3:
            for line, props in zip(self.plane_gdf["geometry"][1:-1], self.plane_geojson["features"][1:-1]):
                line_xy = line.xy
                plt.plot(*line_xy, color='red', linewidth=2, linestyle=":" if props["properties"]["interpolated"] else "-")

        print("first line", plane_first_line_xy)
        start_marker = mpl.path.Path([[-2.5,4],[0,-4],[2.5,4]],[1,2,2])
        
        
        start_rotation = bearing(
            *transform(Proj(init=f"EPSG:{self.crs}") , Proj(init='EPSG:4326'), plane_first_line_xy[0][0], plane_first_line_xy[1][0]),
            *transform(Proj(init=f"EPSG:{self.crs}") , Proj(init='EPSG:4326'), plane_first_line_xy[0][1], plane_first_line_xy[1][1]))
        # marker at rotation zero points downwards; 90 points east, 180 points up; -90, 270 points west
        # which is weird, since it's counterclockwise
        # compass_bearings returned by bearings are clockwise. 0: north; 180: south; 90: east; -90: west. 
        # so we have to multiply by -1 to reflect across the vertical axis
        # and add 180 to reflect across the horizontal axis.
        # phew!
        start_marker = start_marker.transformed(mpl.transforms.Affine2D().rotate_deg(180 + (-1 * start_rotation)))
        print("start_marker", start_rotation)
        plt.plot(*plane_first_line_xy, color='red', linewidth=2, 
            marker=start_marker, markeredgecolor='red', markerfacecolor='red', markersize=10, markevery=[0],
            linestyle=":" if plane_first_line_props["properties"]["interpolated"] else "-"
            )

        # TODO erase this crap.
        # TODO: to get the arrows positioned right
        # (a) verify that the sort order is correct with an assertion
        # (b) check the first/last points on the map with https://epsg.io/transform#s_srs=2263&t_srs=4326&x=1008085.5476291947&y=188163.5085292824
        # first line (array('d', [1008085.5476291947, 1008321.5125119365, 1013809.5113116304]), array('d', [188163.5085292824, 187945.14369742604, 174944.67593990226]))
        #                                                               40°40'59.196",-73°54'50.616"
        #                                                               Bushwick (near Chauncey J)
        # last line (array('d', [972081.46942511, 969538.8233419291, 969455.6974232981, 969015.899166967, 972815.8461703423, 991687.3662600536]), array('d', [180340.01645021865, 187369.31956139568, 187504.17174576147, 189420.81446445087, 190344.12501779612, 237239.45081197194]))
        #                                                               https://epsg.io/transform#s_srs=2263&t_srs=4326&x=1013809.5113116304&y=174944.67593990226
        #                                                               40°38'48.516",-73°53'36.528"
        #                                                               Canarsie near the pier and the end of hte L, but closer to Fresh Creek
        # (c) futz with rotations.
        
        # start_marker  -22.737663038347076 and we want it to be facing NNW (so like maybe -25?)
        # end_marker 160.15010548121617 and we want it to be facing SSE (so like 154)


        print("last line", plane_last_line_xy)
        end_marker = mpl.path.Path([[-2.5,4],[0,-4],[2.5,4]],[1,2,2])
        end_rotation = bearing(
            *transform(Proj(init=f"EPSG:{self.crs}") , Proj(init='EPSG:4326'), plane_last_line_xy[0][-2], plane_last_line_xy[1][-2]),
            *transform(Proj(init=f"EPSG:{self.crs}") , Proj(init='EPSG:4326'), plane_last_line_xy[0][-1], plane_last_line_xy[1][-1]))
        end_marker = end_marker.transformed(mpl.transforms.Affine2D().rotate_deg(180 +  (-1 * end_rotation)))
        print("end_marker", end_rotation)
        plt.plot(*plane_last_line_xy, color='red', linewidth=2, 
            marker=end_marker, markeredgecolor='red', markerfacecolor='red', markersize=10, markevery=[-1],
            linestyle=":" if plane_last_line_props["properties"]["interpolated"] else "-"
            )

        # labels
        if include_background:
            # end of traj
            if self.end_time.day != self.start_time.day:
                end_time = self.end_time.strftime("%-I:%M %p %-m/%d/%Y")    # TODO these may be in UTC.
            else:
                end_time = self.end_time.strftime("%-I:%M %p")              # TODO these may be in UTC.

            first_point = (plane_first_line_xy[0][0], plane_first_line_xy[1][0]) 
            last_point = (plane_last_line_xy[0][-1], plane_last_line_xy[1][-1]) 

            # TODO: use xytext and textcoords for offset from the point
            ax.text(last_point[0], last_point[1], f"{self.nnum} {end_time}",
                    # bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10}
                    )

            # start of traj
            start_time = self.start_time.strftime("%-I:%M %p %-m/%d/%Y") # TODO these may be in UTC.
            ax.text(first_point[0], first_point[1], f"{self.nnum} {start_time}",
                    # bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10}
                    )

        ax.xaxis.set_visible(False) 
        ax.yaxis.set_visible(False) 

        if fn:
            self.custom_map_fn = fn
        map_fn = self.get_map_fn()

        fig.savefig(map_fn, bbox_inches = 'tight',
            pad_inches = 0, 
            facecolor='black'
        )

        return map_fn, plt
        # TODO experiment with this.
        # gca().set_axis_off()
        # subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
        #             hspace = 0, wspace = 0)
        # margins(0,0)
        # gca().xaxis.set_major_locator(NullLocator())
        # gca().yaxis.set_major_locator(NullLocator())


class FlightpathShingle(Flightpath):
    def __init__(self, parent_flightpath, trajectory_points):
        print(trajectory_points[0]["corrected_time"], trajectory_points[-1]["corrected_time"])
        
        # trajectory_points is MOST RECENT FIRST
        assert len(trajectory_points) == 1 or trajectory_points[0]["corrected_time"] <= trajectory_points[1]["corrected_time"]
        if len(trajectory_points) == 0 or (trajectory_points[-1]['corrected_time'] - trajectory_points[0]['corrected_time']).total_seconds() < SHINGLE_DURATION:
            raise HelicopterShinglingError("trajectory duration is too short to generate shingles" )
        if len(trajectory_points) < MAP_MIN_POINTS:
            raise HelicopterShinglingError("shingle has too few points")


        super().__init__(parent_flightpath.icao_hex, nnum=parent_flightpath.nnum, start_time=trajectory_points[0]["corrected_time"], end_time=trajectory_points[-1]["corrected_time"], crs=parent_flightpath.crs)

        self.is_hovering = None # this is special to FlightpathShingle

        self.trajectory_points = trajectory_points
        trajectory_points_grouped = group_rows_between_gaps(trajectory_points)
        plane, plane_geojson = rows_to_geojson(trajectory_points_grouped)
        self.points_cnt = len(trajectory_points)
        self.groups_cnt = len(trajectory_points_grouped)
        if self.crs != DEFAULT_CRS:
            plane = plane.to_crs(epsg=self.crs)
        self.plane_gdf = plane
        self.plane_geojson = plane_geojson


class NeighborhoodsCounter():
    def __init__(self, neighborhood_shapefiles_and_name_columns, neighborhood_name_transforms={}, crs=None):
        """
        neighborhood_shapefiles_and_name_columns should be a list of (geojson_filename, name_column), e.g. [(nynta.geojson, "NTAName")]
        """
        self.gdfs = [load_json_to_gdf(shp_fn, crs=crs) for shp_fn, name in neighborhood_shapefiles_and_name_columns]
        self.gdf_name_cols = [name for shp_fn, name in neighborhood_shapefiles_and_name_columns]
        self.neighborhood_name_transforms = neighborhood_name_transforms

    def neighborhoods_under_lines(self, lines_gdf):
        counter = Counter()
        for neighborhoods, name_col in zip(self.gdfs, self.gdf_name_cols):
            # count points per neighborhood
            res = gpd.tools.sjoin(neighborhoods, lines_gdf, how="inner")
            counter += Counter(res[name_col])
        return [self.neighborhood_name_transforms.get(n, n) for n, cnt in counter.most_common() if self.neighborhood_name_transforms.get(n, n)]

    def neighborhoods_under_points(self, points_gdf):
        counter = Counter()
        for neighborhoods, name_col in zip(self.gdfs, self.gdf_name_cols):
            # count points per neighborhood
            res = gpd.tools.sjoin(neighborhoods, points_gdf, how="inner")
            counter += Counter(res[name_col])
        return [self.neighborhood_name_transforms.get(n, n) for n, cnt in counter.most_common() if self.neighborhood_name_transforms.get(n, n)]

try:
    pwd = dirname(abspath(__file__)) or getcwd()
except:
    pwd = ''
