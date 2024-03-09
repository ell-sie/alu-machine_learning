#!/usr/bin/env python3
""" Script for getting user location"""
import sys
import requests
import time


if __name__ == '__main__':
    url = sys.argv[1]
    response = requests.get(url)
    res = response.json()

    if response.status_code == 200:
        print(res['location'])
    elif response.status_code == 404:
        print('Not found')
    elif response.status_code == 403:
        limit = int(response.headers['X-Ratelimit-Reset'])
        start = int(time.time())
        elapsed = int((limit - start) / 60)
        print('Reset in {} min'.format(int(elapsed)))
