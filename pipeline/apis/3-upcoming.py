#!/usr/bin/env python3
"""
this displays the upcoming launch information.
"""

import requests
from datetime import datetime, timezone

def get_local_time(utc_time):
    """
this displays the upcoming launch information.
"""
    utc_time = datetime.fromisoformat(utc_time.replace("Z", "+00:00"))
    local_time = utc_time.replace(tzinfo=timezone.utc).astimezone(tz=None)
    return local_time.strftime("%Y-%m-%dT%H:%M:%S")

if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/launches/upcoming'
    launches = requests.get(url)
    
    if launches.status_code == 200:
        launches = launches.json()
        
        # Sort the list of upcoming launches by their dates
        launches.sort(key=lambda x: x['date_unix'])
        
        # Extract information about the soonest upcoming launch
        soonest_launch = launches[0]
        launch_name = soonest_launch['name']
        launch_date_utc = soonest_launch['date_local']
        rocket_id = soonest_launch['rocket']
        launchpad_id = soonest_launch['launchpad']
        
        # Fetch rocket information
        rocket_url = 'https://api.spacexdata.com/v4/rockets/{}'.format(rocket_id)
        rocket_data = requests.get(rocket_url).json()
        rocket_name = rocket_data['name']
        
        # Fetch launchpad information
        launchpad_url = 'https://api.spacexdata.com/v4/launchpads/{}'.format(launchpad_id)
        launchpad_data = requests.get(launchpad_url).json()
        launchpad_name = launchpad_data['name']
        launchpad_locality = launchpad_data['locality']
        
        # Convert launch date to local time
        launch_date_local = get_local_time(launch_date_utc)
        
        # Print the information
        print('{} ({}) {} - {} ({})'.format(launch_name, launch_date_local, rocket_name, launchpad_name, launchpad_locality))
    else:
        print("Failed to fetch upcoming launches data")
