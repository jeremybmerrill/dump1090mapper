import unittest
import datetime 
from os.path import join, abspath, dirname
import dump1090mapper.flightpath as flightpath

class TestFlightpathMethods(unittest.TestCase):


    def test_jsonification(self):
        fp = flightpath.Flightpath.from_json("N917PD-2019-07-31T09_17_45-2019-07-31T09_20_44.flightpath.json")
        fp.to_json(fn="/tmp/flightpathtest.json")
        with open("/tmp/flightpathtest.json") as f:
            output = f.read()
        with open("N917PD-2019-07-31T09_17_45-2019-07-31T09_20_44.flightpath.json") as f:
            input_ = f.read()
        self.assertEqual(output, input_)

    def test_neighborhoods_underneath(self):
        fp = flightpath.Flightpath.from_json("N917PD-2019-07-31T09_17_45-2019-07-31T09_20_44.flightpath.json")
        self.assertEqual(fp.crs, 2263)
        self.assertEqual(fp.neighborhoods_underneath(), 
            ['Williamsburg Brg',
             'Manhattan Brg',
             'DUMBO-Vinegar Hill-Downtown Brooklyn-Boerum Hill',
             'Lower East Side',
             'Brooklyn Heights-Cobble Hill'])
if __name__ == '__main__':
    unittest.main()
