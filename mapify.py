#!/usr/bin/env python

import argparse 
from dump1090mapper.flightpath import Flightpath, NeighborhoodsCounter, COPTERS, HelicopterShinglingError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n-number', 
                        help='<N12345> aircraft N-Number (or other label)')
    parser.add_argument('-i', '--icao-hex', 
                        help='<ABCDEF> the hex ICAO address of the aircraft')
    parser.add_argument('-s', '--start-time', 
                        help='start time for trajectory to be mapped (must also supply end-time), e.g. 2019-04-02 04:37:38)')
    parser.add_argument('-e', '--end-time', 
                        help='end time for trajectory to be mapped (must also supply start-time, e.g. 2019-04-02 04:37:38)')
    parser.add_argument('-a', '--arbitrary-marker', 
                        help='<lon,lat> arbitrary point to be mapped, in lon,lat format, e.g. -73.9037267,40.708143')
    parser.add_argument('-b', '--exclude-background', action="store_true",
                        help='Exclude the map background/labels; show just trajectory')
    args = parser.parse_args()


    assert args.n_number or args.icao_hex
    crs = 2263 # NY State Plane                                                                 
    aircraft_nnum = args.n_number
    include_background = not args.exclude_background
    arbitrary_marker = [float(c) for c in args.arbitrary_marker.split(",")] if args.arbitrary_marker else None 
    icao_hex = args.icao_hex or COPTERS.get(aircraft_nnum, None)
    assert icao_hex is not None

    flightpath = Flightpath(icao_hex, aircraft_nnum, crs=crs) #  (datetime(2019, 7, 27, 6, 8, 15), datetime(2019, 7, 27, 6, 44, 45)
    flightpath = Flightpath.from_json('N917PD-2019-08-06T00_09_23-2019-08-06T00_37_54.flightpath.json')
    try:
        [s for s in flightpath.as_shingles()]
    except HelicopterShinglingError:
        []

    map_fn, plt = flightpath.to_map(background_color='#ADD8E6')
    fp_json = flightpath.to_json()
    with open(flightpath.json_fn(), 'w') as f:
        f.write(fp_json)

    # plt.show(block=True)