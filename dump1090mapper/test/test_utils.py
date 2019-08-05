import unittest
import datetime 
from os.path import join, abspath, dirname
import dump1090mapper.utils as utils
class TestUtilsMethods(unittest.TestCase):

# split_rows_at_timediff
# rows_to_geojson

# point_at_distance
# bearing

    # The latitude is 40.730610, and the longitude is -73.935242
    def test_haversine(self):
        self.assertGreaterEqual(utils.haversine(-73.9755762,40.7623904,-74.000844,40.7158321), 3.47)
        self.assertLessEqual(utils.haversine(-73.9755762,40.7623904,-74.000844,40.7158321), 3.49)


    def test_hour_differences(self):
        with open( join(dirname(abspath(__file__)),    "rows.python"), 'r') as f:
            rows = eval(f.read())
        self.assertEqual(utils._hour_differences(rows), {1: -19.0, 2: -5.0, 3: 0})


    def test_group_rows_between_gaps(self):
        for fn in ["trajectory_points.python", "trajectory_points2.python", "trajectory_points3.python"]:
            with open( join(dirname(abspath(__file__)),    fn), 'r') as f:
                rows = eval(f.read())
            grouped_rows = utils.group_rows_between_gaps(rows)
            for group in grouped_rows:
                self.assertGreaterEqual(len(group), 2)
                for (a, b) in zip(group[:-1], group[1:]):
                    self.assertGreaterEqual(a.get("corrected_time",a["datetz"]) , b.get("corrected_time", b["datetz"]) )

            for (a,b) in zip(grouped_rows[:-1], grouped_rows[1:]):
                self.assertGreaterEqual(a[0].get("corrected_time", a[0]["datetz"]) , b[0].get("corrected_time", b[0]["datetz"]) )

            for group in grouped_rows[1:]:
                self.assertGreaterEqual(b[0]["timediff"], utils.MAX_UNMARKED_INTERPOLATION_SECS)

    def test_bearing(self):
        # white house                  38.8996473,-77.0346165
        # scott circle, DC (due north) 38.9078277,-77.0346165
        # bearing is zero
        self.assertAlmostEqual(utils.bearing(-77.0346165,38.8996473 , -77.0346165, 38.9078277), 0.0, delta=1) # north
        self.assertAlmostEqual(utils.bearing(-77.0346165,38.9078277, -77.0346165, 38.8996473), 180.0, delta=1) # south
        # lincoln memorial: 38.8893709,-77.0503125
        # capitol:          38.8897441,-77.0093534,19.34z
        self.assertAlmostEqual(utils.bearing(-77.0503,38.889, -77.0093, 38.889), 90.0, delta=1) # east
        self.assertAlmostEqual(utils.bearing(-77.0093,38.889, -77.0503, 38.889), -90.0, delta=1) # west

if __name__ == '__main__':
    unittest.main()
