from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic, get_body, solar_system_ephemeris
from astropy.time import Time
from skyfield.api import load, utc
import numpy as np
import datetime

def get_celestial_positions(time_arr):

    ts = load.timescale()

    # Convert np.datetime64 array to Skyfield Time
    py_datetimes = [t.astype('O').replace(tzinfo=utc) for t in time_arr]  # Convert to Python datetime objects
    sky_time = ts.utc(py_datetimes)

    # Load ephemeris
    eph = load('de421.bsp')
    earth, sun, moon = eph['earth'], eph['sun'], eph['moon']

    # Observe Sun and Moon from Earth for all times at once
    astrometric_sun = earth.at(sky_time).observe(sun).apparent()
    astrometric_moon = earth.at(sky_time).observe(moon).apparent()

    # Get RA, Dec
    # ra_sun, dec_sun, _ = astrometric_sun.radec()
    # ra_moon, dec_moon, _ = astrometric_moon.radec()

    # # Apply any empirical offsets (adjust if necessary)
    # solar = np.column_stack([
    #     ra_sun.hours - 1/60,       # Subtract 1 minute in RA
    #     dec_sun.degrees + 4/60     # Add 4 arcminutes in Dec
    # ])
    # lunar = np.column_stack([
    #     ra_moon.hours - 8/60,      # Subtract 8 minutes in RA
    #     dec_moon.degrees           # Keep Dec as is
    # ])

    return astrometric_sun, astrometric_moon

def get_sun_hecl(time_arr, ra_list, dec_list):
    """
    Compute heliocentric ecliptic longitude separation and ecliptic latitude
    for multiple times and sky directions.

    Parameters:
        time_arr (np.ndarray): Array of np.datetime64 objects
        ra_list (list or np.ndarray): Array of RA values in degrees
        dec_list (list or np.ndarray): Array of Dec values in degrees

    Returns:
        sun_hecl_arr (2D np.ndarray): shape (N_time, N_dir) for longitude separation [deg]
        sun_beta_arr (2D np.ndarray): shape (N_time, N_dir) for ecliptic latitude [deg]
    """

    time_astropy = Time(time_arr)

    with solar_system_ephemeris.set('de440'):
        # Get Sun position at each time in ecliptic coordinates
        sun_co = get_body('sun',time_astropy)
        sun_coords = sun_co.transform_to(GeocentricTrueEcliptic(equinox=time_astropy))
        # Get Moon position at each time in ecliptic coordinates
        moon_coords = get_body('moon', time_astropy)

    # ra_list = np.asarray(ra_list)
    # dec_list = np.asarray(dec_list)

    # N_time = len(time_astropy)
    # N_dir = len(ra_list)

    # sun_hecl_arr = np.zeros(N_time)
    # sun_beta_arr = np.zeros(N_time)

    target_ecl = SkyCoord(ra=ra_list, dec=dec_list, unit='deg', frame='icrs').transform_to(GeocentricTrueEcliptic(equinox=time_astropy))
    delta_long = np.abs(target_ecl.lon.deg - sun_coords.lon.deg)
    sun_hecl_arr = np.where(delta_long > 180, 360 - delta_long, delta_long)
    sun_beta_arr = np.abs(target_ecl.lat.deg)
        

    return sun_hecl_arr, sun_beta_arr, sun_co, moon_coords

import numpy as np

def zod_dist_read(zod_file, zod_xsize=10, zod_ysize=19):
    """
    Reads the zodiacal light distribution from a file.

    Parameters:
        zod_file (str): Path to the file
        zod_xsize (int): Number of longitude grid points (default 36)
        zod_ysize (int): Number of latitude grid points (default 19)

    Returns:
        arr (2D np.ndarray): Intensity values [zod_ysize x zod_xsize]
        hecl_lon (1D np.ndarray): Heliocentric ecliptic longitudes [zod_xsize]
        hecl_lat (1D np.ndarray): Heliocentric ecliptic latitudes [zod_ysize]
    """

    with open(zod_file, 'r') as f:
        # Skip first two header lines
        _ = f.readline()
        _ = f.readline()

        # Read dummy float
        _ = float(f.readline().strip())

        # Read heliocentric latitudes
        hecl_lat = np.array([float(f.readline().strip()) for _ in range(zod_ysize)])

        hecl_lon = np.zeros(zod_xsize)
        arr = np.zeros((zod_ysize, zod_xsize))

        for ix in range(zod_xsize):
            parts = f.readline().strip().split()
            # First element is longitude
            hecl_lon[ix] = float(parts[0])
            # Rest are intensity values (1D row of latitudes)
            for iy in range(zod_ysize):
                if iy == 0 and len(parts) > 1:
                    arr[iy, ix] = float(parts[1])
                else:
                    arr[iy, ix] = float(f.readline().strip())

    return arr, hecl_lon, hecl_lat


if __name__ == "__main__":

    # Define time array (every 600s for 1 hour)
    time_start = np.datetime64(datetime.datetime.now())
    time_end = 3600  # seconds
    dt = 600         # seconds
    end = np.arange(0.0, time_end, dt)
    time_arr = time_start + end.astype('timedelta64[s]')

    # Define target directions
    ra_list = (np.random.random(len(time_arr)))*360  # degrees
    dec_list = np.random.random(len(time_arr))*180 - 90 

    sun_hecl, sun_beta, sun, moon = get_sun_hecl(time_arr, ra_list, dec_list)

    # Print results
    for i, t in enumerate(time_arr):
        print(f"\nTime: {t}, Solar Possition: {sun[i].ra.hour:.4f} hrs, {sun[i].dec.deg:.4f} deg, , Lunar Possition: {moon[i].ra.hour:.4f} hrs, {moon[i].dec.deg:.4f} deg")
        print(f"  RA: {ra_list[i]}, Dec: {dec_list[i]} -> Hecl: {sun_hecl[i]:.2f} deg, Beta: {sun_beta[i]:.2f} deg")
