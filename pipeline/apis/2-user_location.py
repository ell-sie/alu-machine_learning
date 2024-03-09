#!/usr/bin/env python3
"""
Prints the location of a specific Github user.
"""
import requests
import sys
import time

def get_user_location(user_url):
    headers = {'Accept': 'application/vnd.github.v3+json'}
    r = requests.get(user_url, headers=headers)

    if r.status_code == 200:
        return r.json().get('location', 'Location not available')
    elif r.status_code == 404:
        return "Not found"
    elif r.status_code == 403:
        rate_limit = int(r.headers['X-Ratelimit-Reset'])
        now = int(time.time())
        minutes = int((rate_limit - now) / 60)
        return f"Reset in {minutes} min"
    else:
        return f"Unexpected status code: {r.status_code}"

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <user_api_url>")
        sys.exit(1)

    user_url = sys.argv[1]
    location = get_user_location(user_url)
    print(location)
