#!/usr/bin/env python3
"""
Prints the location of a specific Github user
"""
import sys
import requests
import time

def get_user_location(api_url):
    """
    Prints the location of a specific GitHub user.
    """
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            user_data = response.json()
            location = user_data.get("location")
            if location:
                print(f"{user_data['login']}'s location: {location}")
            else:
                print(f"{user_data['login']} does not have a specified location.")
        elif response.status_code == 404:
            print("Not found")
        elif response.status_code == 403:
            reset_time = int(response.headers.get("X-Ratelimit-Reset"))
            current_time = int(time.time())
            time_remaining = max(0, reset_time - current_time)
            print(f"Reset in {time_remaining // 60} min")
        else:
            print(f"Error fetching data. Status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 2-user_location.py <GitHub_API_URL>")
    else:
        github_api_url = sys.argv[1]
        get_user_location(github_api_url)
