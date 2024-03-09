#!/usr/bin/env python3
""" Script that displays the number of launches per rocket"""
import requests

def get_launch_data():
    """ Script that displays the number of launches per rocket"""
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)
    return response.json()

def count_launches_by_rocket(launch_data):
    """ Script that displays the number of launches per rocket"""
    rocket_launches = {}
    for launch in launch_data:
        rocket_name = launch["rocket"]
        if rocket_name in rocket_launches:
            rocket_launches[rocket_name] += 1
        else:
            rocket_launches[rocket_name] = 1
    return rocket_launches

def main():
    launch_data = get_launch_data()
    rocket_launches = count_launches_by_rocket(launch_data)

    # Sort by number of launches (descending) and then alphabetically
    sorted_launches = sorted(rocket_launches.items(), key=lambda x: (-x[1], x[0]))

    for rocket, launches in sorted_launches:
        print(f"{rocket}: {launches}")

if __name__ == "__main__":
    main()
