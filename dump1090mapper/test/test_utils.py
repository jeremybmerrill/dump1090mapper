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
        with open( join(dirname(abspath(__file__)),    "trajectory_points.python"), 'r') as f:
            rows = eval(f.read())
        grouped_rows = utils.group_rows_between_gaps(rows)
        for group in grouped_rows:
            self.assertGreaterEqual(len(group), 2)
            for (a, b) in zip(group[:-1], group[1:]):
                self.assertGreaterEqual(a["datetz"], b["datetz"])

        for (a,b) in zip(grouped_rows[:-1], grouped_rows[1:]):
            self.assertGreaterEqual(a[0]["datetz"], b[0]["datetz"])

        for group in grouped_rows[1:]:
            self.assertGreaterEqual(b[0]["timediff"], MAX_UNMARKED_INTERPOLATION_SECS)



if __name__ == '__main__':
    unittest.main()
