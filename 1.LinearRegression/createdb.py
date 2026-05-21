import numpy as np
import pandas as pd
import requests

# Export Ranks Avg:
    # filter by only 333 event ids (only 3x3 solvers)
# ping the WCA API
    # get all of the solves for a given person

def get_WCA_profile(WCA_id: str) -> dict:
    # base URL for WCA API
    WCA_BASE = 'https://www.worldcubeassociation.org/api/v0'
    # obtain competition results for a person given a specific WCA ID
    
    # profile with all stats
    profile = {}
    
    person_info = requests.get(f'{WCA_BASE}/persons/{WCA_id}').json()
    # dict_keys(['person', 'competition_count', 'personal_records', 'medals', 'records', 'total_solves'])
    
    profile['WCA_ID'] = WCA_id
    profile['best_single'] = person_info['personal_records']['333']['single']['best'] / 100
    profile['best_avg'] = person_info['personal_records']['333']['average']['best'] / 100
    
    comp_results = requests.get(f'{WCA_BASE}/persons/{WCA_id}/results').json()
    
    # all of their results
    results = []
    
    # all competition results
    for r in comp_results:
        # only filter for 3x3 solves in the first round with completed averages
        if r['event_id'] == '333' and r['round_type_id'] == '1' and r['average'] > 0:
            # results from a given competition
            res = {}
            
            # competition entries for solve ID and the attempts, best time, worst time, average of 3 (ao3)
            # convert from centiseconds to seconds by dividing by 100
            solves = list(map(lambda x: x/100, r['attempts']))
            
            # negative times represent DNS (did not solve) or DNF (did not finish)
            # if there are more than 1 of either a DNS or DNF, the average should not be counted
            if len(list(filter(lambda x: x < 0, solves))) > 1: continue
            
            solves.sort()
            
            # shortest time is the best time, unless there is a DNF/DNS, then it is the second shortest time
            best = solves[0] if solves[0] > 0 else solves[1]
            # the worst time is the highest time unless it is a DNS or DNF, in which case it is that
            worst = solves[-1] if solves[0] > 0 else solves[0]
            ao3 = sum(solves[1:4]) / 3
            
            res = {
                'best': best,
                'worst': worst,
                'ao3': ao3,
                'solves': solves
            }
            
            comp_entry = {r['id'] : res}
            results.append(comp_entry)
    
    profile['results'] = results
    
    return profile

print(get_WCA_profile('2009ZEMD01'))