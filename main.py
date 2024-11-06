import pandas as pd
import random 
import copy
import json
from openai import AsyncOpenAI, APITimeoutError
import concurrent.futures
from collections import defaultdict
from tabulate import tabulate
import matplotlib.pyplot as plt
import plotille
import time
import asyncio
candidates_file = "processed_trial_data_anonymized.csv"
async_client = AsyncOpenAI()

def zero_shot_no_reasoning(role, candidateA_transcript, candidateB_transcript, candidateA_resume, candidateB_resume):
    system_prompt = f"""
    You are provided the interviews and resumes of candidates we're considering hiring for the role of {role}. Identify the stronger candidate for the role based on their interview performance and the strength of their resumes. Evaluate the strengths and weaknesses of each candidate, then pick which one that you prefer.

        Interview of Candidate A: {candidateA_transcript}
        Interview of Candidate B: {candidateB_transcript}

        Resume of Candidate A: {candidateA_resume}
        Resume of Candidate B: {candidateB_resume}

    You must output in the following format which matches that of a JSON and I want nothing else:
    {{
        "winner": "",
    }}
    Ensure the 'winner' field is either A or B, it should be A if candidate A was better and B if candidate B was better.
    """

    return system_prompt

async def execute_comparison_async(role, candidateA_transcript, candidateB_transcript, candidateA_resume, candidateB_resume):
    system_prompt = zero_shot_no_reasoning(role, candidateA_transcript, candidateB_transcript, candidateA_resume, candidateB_resume)
    try:
        completion = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt}
            ],
            timeout=8
        )
        try:
            response_json = json.loads(completion.choices[0].message.content)
            winner = response_json["winner"]
            return winner
        except json.JSONDecodeError as e:
            result = random.choice(["A", "B"])
            return result
    except (APITimeoutError, Exception) as e:
        print("Ever timing out")
        return random.choice(["A", "B"])

def range_inclusive_randomized(lower_bound, higher_bound):
    numbers = list(range(lower_bound, higher_bound + 1))
    random.shuffle(numbers)
    return numbers

def comparison(candidateA, candidateB):

    return candidateA < candidateB

def simulateTournament(candidates):

    running_list = []
    ordering = []
    while True:
        running_list = []
        temp = []
        for i in range(0, len(candidates), 2):
            if i == len(candidates) - 1:
                running_list.append(candidates[i])
                break
            smaller = comparison(candidates[i], candidates[i + 1])

            if smaller:
                temp.append(candidates[i + 1])
                running_list.append(candidates[i])
            else:
                temp.append(candidates[i])
                running_list.append(candidates[i + 1])

            if len(temp) == 2:
                smaller = comparison(temp[0], temp[1])
                if smaller:
                    ordering.append(temp[1])
                    ordering.append(temp[0])
                else:
                    ordering.append(temp[0])
                    ordering.append(temp[1])
                
                temp = []
            
        if len(temp) != 0:
            ordering.append(temp[0])

        if len(running_list) == 1:
            ordering.append(running_list[0])
            break
        else:
            candidates = copy.deepcopy(running_list)

    ordering.reverse()
    return ordering

async def trueTournament(role, candidates):
    ordering = []
    while True:
        running_list = pd.DataFrame(columns=candidates.columns)
        tasks = []
        pairs = []
        
        for i in range(0, len(candidates), 2):
            if i == len(candidates) - 1:
                running_list = pd.concat([running_list, candidates.iloc[[i]]], ignore_index=True)
                continue
                
            task = execute_comparison_async(
                role,
                candidates.iloc[i, 2],
                candidates.iloc[i + 1, 2],
                candidates.iloc[i, 1],
                candidates.iloc[i + 1, 1]
            )
            tasks.append(task)
            pairs.append((i, i + 1))
        
        if tasks:
            winners = await asyncio.gather(*tasks)
            
            for (i, i_plus_1), winner in zip(pairs, winners):
                if winner == "A":
                    ordering.append(candidates.iloc[i_plus_1, 0])
                    running_list = pd.concat([running_list, candidates.iloc[[i]]], ignore_index=True)
                else:
                    ordering.append(candidates.iloc[i, 0])
                    running_list = pd.concat([running_list, candidates.iloc[[i_plus_1]]], ignore_index=True)
        
        if len(running_list) == 1:
            ordering.append(running_list.iloc[0, 0])
            break
        else:
            candidates = running_list
    
    ordering.reverse()
    return ordering

async def run_multiple_tournaments(role, candidates, num_tournaments=5):
    placement_tracker = defaultdict(list)
    
    tournament_tasks = []
    for _ in range(num_tournaments):
        shuffled_candidates = candidates.sample(frac=1).reset_index(drop=True)
        tournament_tasks.append(trueTournament(role, shuffled_candidates))
    
    tournament_results = await asyncio.gather(*tournament_tasks)
    
    for tournament_ranking in tournament_results:
        for place, candidate_id in enumerate(tournament_ranking):
            placement_tracker[candidate_id].append(place + 1)
    
    average_placements = {
        candidate_id: sum(placements) / len(placements)
        for candidate_id, placements in placement_tracker.items()
    }
    
    final_ranking = sorted(average_placements.items(), key=lambda x: x[1])
    
    detailed_results = []
    for candidate_id, avg_placement in final_ranking:
        placements = placement_tracker[candidate_id]
        detailed_results.append({
            'candidate_id': candidate_id,
            'average_placement': avg_placement,
            'best_placement': min(placements),
            'worst_placement': max(placements),
            'placement_variance': sum((x - avg_placement) ** 2 for x in placements) / len(placements),
            'all_placements': placements
        })
    
    return final_ranking, detailed_results

def start_multiple_tournaments(role, candidates, num_tournaments=5):
    return asyncio.run(run_multiple_tournaments(role, candidates, num_tournaments))

def top_k(k, candidates):

    num_top = 0

    for i in range(0, k):

        if candidates[i] <= k:
            num_top += 1
    
    return num_top / k


def simulate_synthetic(k, bottom_range, top_range):

    sum_1_percent = 0
    sum_10_percent = 0
    sum_25_percent = 0
    sum_50_percent = 0
    sum_75_percent = 0
    sum_100_percent = 0

    one_top = (top_range - bottom_range + 1) // 100
    ten_top = (top_range - bottom_range + 1) // 10
    twenty_five_top = (top_range - bottom_range + 1) // 4
    fify_top = (top_range - bottom_range + 1) // 2
    seventy_five_top = ((top_range - bottom_range + 1) * 3 ) // 4
    hundred_top = (top_range - bottom_range + 1)
    for i in range(k):
        place_counter = defaultdict(int)
        for j in range(7):
            candidates = range_inclusive_randomized(bottom_range, top_range)
            result = simulateTournament(candidates)

            for i, val in enumerate(result):
                place_counter[val] += (i + 1)

        sorted_list = []

        for i, key in enumerate(place_counter):
            sorted_list.append((place_counter[key] / 7, key))
        sorted_list.sort()
        result = []
        
        for val in sorted_list:
            result.append(val[1])

        one = top_k(one_top, result)
        ten = top_k(ten_top, result)
        twenty_five = top_k(twenty_five_top, result)
        fifty = top_k(fify_top, result)
        seventy_five = top_k(seventy_five_top, result)
        hundred = top_k(hundred_top, result)

        sum_1_percent += one
        sum_10_percent += ten
        sum_25_percent += twenty_five
        sum_50_percent += fifty
        sum_75_percent += seventy_five
        sum_100_percent += hundred

    data = [["Top 1 Percent", sum_1_percent/k], ["Top 10 Percent", sum_10_percent/k], ["Top 25 Percent", sum_25_percent/k], ["Top 50 Percent", sum_50_percent/k], ["Top 75 Percent", sum_75_percent/k], ["Top 100 Percent", sum_100_percent/k]]

    return data

def print_tables_and_graphs(data, headers):

    print(tabulate(data, headers, tablefmt="pretty"))

    x = [1, 10, 25, 50, 75, 100]
    y = [val[1] for val in data]

    print(plotille.plot(x, y, width=40, height=10, x_min=0))

def main():
    data = pd.read_csv(candidates_file)
    
    # Run multiple tournaments and get both rankings and detailed results
    final_ranking, detailed_results = start_multiple_tournaments("Python Developer", data, 1)
    
    print("\nFinal Rankings Based on Average Placement:")
    print(tabulate(
        [(i+1, candidate_id, f"{avg_placement:.2f}") 
         for i, (candidate_id, avg_placement) in enumerate(final_ranking[:20])],
        headers=["Rank", "Candidate ID", "Avg Placement"],
        tablefmt="pretty"
    ))
    
    print("\nDetailed Statistics for Top 10 Candidates:")
    detailed_stats = [(
        i+1,
        result['candidate_id'],
        f"{result['average_placement']:.2f}",
        result['best_placement'],
        result['worst_placement'],
        f"{result['placement_variance']:.2f}"
    ) for i, result in enumerate(detailed_results[:10])]
    
    print(tabulate(
        detailed_stats,
        headers=["Rank", "Candidate ID", "Avg Place", "Best", "Worst", "Variance"],
        tablefmt="pretty"
    ))

    print(final_ranking)

    # # Testing on synthetic data and generating graph
    # data = simulate_synthetic(200, 1, 1000)
    # headers = ["Top K Level", "Accuracy"]
    # print_tables_and_graphs(data, headers)


if __name__ == "__main__":
    main()
