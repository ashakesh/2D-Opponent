"""
From the JSON output of pyinstrument. Calculated some summary statistics and graph.

Usage:
    First train model whilst measuring performance with pyinstrument. For example,
    ```
    pyinstrument -r json -o "flame_data.json" -m fightml.train --dummy-only --save runs/ppo_mix4 --pool runs/pool4 --n-envs 8 --timesteps 30000
    ```
    Note: replacing `json` with `html` outputs data in a pretty and interactive. See pyinstrument documentation for saving json files and other tricks.

    Then get a summary of the training time with the module.
        1. Open python shell (make sure virtual environment is setup before continuing),
        ```
           python 
        ```
        2. import this module.
        ```
            >> from fightml.performance import *
        ```
        3. get summary data from json
        ```
            >> summary_data = summary_times_training("flame_data.json")
        ```
        4. plot summary data as pie chart
        ```
            >> plot_training_times(summary_data, "train_piechart.png")
        ```
"""
import json
from typing import Any
import functools as ft
import matplotlib.pyplot as plt

def summary_times_training(file_path: str):
    f = open(file_path, "r")
    raw_json: str= "\n".join(f.readlines())
    f.close()

    all_data: dict[str, Any] = json.loads(raw_json)
    def concatTimes(t1: dict[str, float], t2: dict[str, float]) -> dict[str,float]:
        t3 : dict[str, float] = {}
        for k1 in t1.keys():
            t3[k1] = t1[k1]
        for k2 in t2.keys():
            if k2 in t3.keys():
                t3[k2] += t2[k2]
            else:
                t3[k2] = t2[k2]
        return t3
    # called in children of function "learn" in "sb3_contrib/ppo_mask/ppo_mask.py"
    def summarise_learn(data: Any) -> dict[str, float]:
        if data["file_path_short"].startswith("game/fightsim"):
            # if from fighting simulation
            return {"fightsim": float(data["time"])}
        elif data["file_path_short"].startswith("torch"):
            #if from torch e.g. when training the model or evaluating the model
            return {"torch": float(data["time"])}
        else:
            # if node is uninteresting
            return ft.reduce(concatTimes, [summarise_learn(child) for child in data["children"]], {})

    def summarise(data: Any):
        if data["function"] == "learn" and data["file_path_short"] == "sb3_contrib/ppo_mask/ppo_mask.py":
            child_data = summarise_learn(data)
            total_learning = float(data["time"]) - sum(child_data.values())
            return concatTimes(child_data, {"Unaccounted training": total_learning})
        else:
            # if node is uninteresting
            return ft.reduce(concatTimes, [summarise(child) for child in data["children"]], {})

    return summarise(all_data["root_frame"])


def plot_training_times(training_summary: dict[str, int], filename: str):
    data = training_summary.items()
    names = [section[0] for section in data]
    times = [section[1] for section in data]
    total_times = sum(times)
    def pie_fmt(x: float):
        return "{:.1f}%\n({:.3f}s)".format(x, x * total_times /100)
    plt.pie(times, labels = names, autopct = pie_fmt)
    plt.title("Time spent during learning (model.learn)")
    plt.savefig(filename)


