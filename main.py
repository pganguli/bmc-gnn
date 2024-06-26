#!/usr/bin/env python3

import argparse
import os
import re
import signal
import subprocess
import time
import pandas as pd
import numpy as np
import pickle

from bmc_gnn.extract_bmc_engine import extract_bmc_engine
from bmc_gnn.extract_frame_time import extract_frame_time
from bmc_gnn.most_similar_circuit import most_similar_circuit
from bmc_gnn.unfold_circuit import unfold_circuit

TIMELIMIT = 1
total_time = 120
FRAME = 2
engine_sequence = {}
start_time = time.time()
end_time = start_time + total_time
D = ""


def extract_data():
    predictors = re.sub(r'[^0-9 \.]', '', D)
    predictors = re.sub(r'\s+', ' ', predictors)
    predictors = re.sub(r'(\d)\. ', r'\1 ', predictors)
    predictors = predictors.strip()
    predictors = predictors.replace(' ', ',')
    predictors = re.split(',', predictors)
    predictors = ([float(num) if '.' in num else int(num) for num in predictors])
    return predictors

def terminate_process(signum, frame):
    """Terminate the process when the time limit is exceeded."""
    print(
        "======================================================================================*****TIMEOUT*****====================================================================================================="
    )
    for x, y in engine_sequence.items():
        print(f"{x}: {y}")
    print(f"\n{D}\n")
    print(
        f"Delay between actual termination time and expected termination time: {round(time.time() - end_time, 2)} secs."
    )
    os._exit(1)  # Forcefully terminate the script


def run_engine(selected_engine, circuit, FLAGS):
    global D
    if selected_engine == "bmcJ":
        command = [
            "abc",
            f"-c read {circuit}; print_stats; &get; bmc3 {FLAGS} -J 2; print_stats",
        ]
    else:
        command = [
            "abc",
            f"-c read {circuit}; print_stats; &get; {selected_engine} {FLAGS}; print_stats",
        ]

    process = subprocess.Popen(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    last_valid_line = None
    output = []
    while True:
        line = process.stdout.readline()
        if not line:
            # print(line)
            break
        line = line.strip()
        #print(line)
        output.append(line)

        if any(line.startswith(x) for x in ["Reached", "Runtime", "No output"]):
            break
        elif re.match(r"^\d+ \+ :", line):
            last_valid_line = line
            D = line
    return last_valid_line


def process_circuits(
    start_time,
    total_time,
    FRAME,
    end_time,
    engine_sequence,
    count,
    s,
    FLAGS,
    CURR_FRAME,
):
    global TIMELIMIT
    cir_name = (args.input_circuit.split("/")[-1]).split(".")[0]
    for file in os.listdir(args.known_circuit_path):
        if time.time() - start_time > total_time:
            break  # Exit the loop if total time has exceeded
        if file.endswith(".aig") and cir_name not in file:
            unfold_circuit(f"{args.known_circuit_path}/{file}", FRAME, args.unfold_path)
    cir = unfold_circuit(args.input_circuit, FRAME, args.unfold_path)
    best_friend = most_similar_circuit(cir, FRAME, args.unfold_path)
    ret = extract_frame_time(FRAME, args.csv_path, best_friend)
    min_time = 4000
    engine = None
    for result in ret:
        if isinstance(result[2][0], str) and result[2][0].isdigit():
            current_time = float(result[2][0])
        elif isinstance(result[2][0], (int, float)):
            current_time = result[2][0]
        else:
            continue

        if current_time < min_time:
            min_time = current_time
            engine = result[0]

    if engine is None:  # No engine gets selected.
        TIMELIMIT = int(end_time - time.time())
        FLAGS = f"-S {FRAME} -T {TIMELIMIT} -F 0 -v"
        engine_sequence[count] = "bmc -g"
        count += 1
        for x, y in engine_sequence.items():
            print(f"{x}: {y}")
        print(
            f"No engine found, hence running bmc3 -g for the remaining time : {TIMELIMIT} secs\n"
        )
        depth_reached = run_engine("bmc3 -g", args.input_circuit, FLAGS)
        if depth_reached is None:
            return FRAME, -1, False, None, None

        if "+" in depth_reached:
            FRAME = int(depth_reached.split(":")[0].strip().split("+")[0].strip())
            print(f"FINAL FRAME REACHED IS {D}\n")
        else:
            for part in depth_reached.split("."):
                if "F =" in part:
                    FRAME = int(part.split("=")[1].strip())
                    print(f"FINAL FRAME REACHED IS {FRAME}\n")
        return FRAME, -1, False, "bmc3 -g", "bmc3g"

    selected_engine = extract_bmc_engine(engine)
    print(
        f"DEPTH ({FRAME}) : DeepGate2 execution + BMC engine selection : {round(time.time() - s, 2)} secs\n"
    )
    engine_sequence[count] = selected_engine
    count += 1
    print(
        f"Outcome at DEPTH ({FRAME}) : Most similar circuit: {best_friend}.aig, Best BMC engine for {best_friend} at Depth {FRAME} : {selected_engine}\n"
    )
    # os.chdir("../../")
    depth_reached = run_engine(selected_engine, args.input_circuit, FLAGS)
    if depth_reached is None:
        print(
            f"Running {selected_engine} on {args.input_circuit.split('/')[-1]} for {TIMELIMIT} second, Depth reached : {FRAME}\n"
        )
        print(f"{D}\n")
        # os.chdir("exp_unf")
        return FRAME, -1, True, selected_engine, engine

    if "+" in depth_reached:
        FRAME = int(depth_reached.split(":")[0].strip().split("+")[0].strip())
    else:
        for part in depth_reached.split("."):
            if "F =" in part:
                FRAME = int(part.split("=")[1].strip())
    print(
        f"Running {selected_engine} on {args.input_circuit.split('/')[-1]} for {TIMELIMIT} second, Depth reached : {FRAME}\n"
    )
    print(f"{D}\n")
    # os.chdir("exp_unf")
    if CURR_FRAME == FRAME:
        return FRAME, -1, True, selected_engine, engine
    return FRAME, FRAME, True, selected_engine, engine


def run_f_mode(args):
    global TIMELIMIT, total_time, FRAME, start_time, end_time
    # circuit = args.input_circuit
    T = TIMELIMIT
    mod_time = T
    first_iter = True
    START_FRAME = 0
    CURR_FRAME = 0
    count = 1
    first_simcheck_done = False
    selected_engine = None
    print(
        f"\nTotal execution time: {total_time} seconds (Start to End of the Framework)\n"
    )
    signal.signal(
        signal.SIGALRM, terminate_process
    )  # Set the signal handler for SIGALRM
    signal.alarm(total_time)  # Schedule an alarm for the total execution time
    try:
        while time.time() - start_time <= total_time:
            s = time.time()
            print(
                "============================================================================================================================================================================================================\n"
            )
            if (
                (CURR_FRAME != START_FRAME and START_FRAME != -1)
                or first_iter
                or START_FRAME != -1
            ):
                first_iter = False
                CURR_FRAME = START_FRAME
                FLAGS = f"-S {START_FRAME} -T {TIMELIMIT} -F 0 -v"
                if first_simcheck_done:
                    last_depth_data = extract_data()
                    time_wasted = round(mod_time - float(last_depth_data[7]),2)
                    print(f'Time wasted in previous iteration is : {time_wasted} sec')
                    mod_time = 2 * mod_time
                    FLAGS = f"-S {START_FRAME} -T {mod_time} -F 0 -v"
                    # print(os.getcwd())
                    # os.chdir('../')
                    print(f"\nStarting at DEPTH ({START_FRAME}) : \n")
                    depth_reached = run_engine(
                        selected_engine, args.input_circuit, FLAGS
                    )
                    if depth_reached is None:
                        print(
                            f"Running {selected_engine} on {args.input_circuit.split('/')[-1]} for {mod_time} second, Depth reached : {FRAME}\n"
                        )
                        print(f"{D}\n")
                        START_FRAME = -1
                        # os.chdir('exp_unf')
                        first_simcheck_done = True
                        continue
                    if "+" in depth_reached:
                        FRAME = int(
                            depth_reached.split(":")[0].strip().split("+")[0].strip()
                        )
                    else:
                        for part in depth_reached.split("."):
                            if "F =" in part:
                                FRAME = int(part.split("=")[1].strip())
                    START_FRAME = FRAME
                    if CURR_FRAME == START_FRAME:
                        START_FRAME = -1
                    print(
                        f"Running {selected_engine} on {args.input_circuit.split('/')[-1]} for {mod_time} second, Depth reached : {FRAME}\n"
                    )
                    print(f"{D}\n")
                    first_simcheck_done = True
                    # os.chdir('exp_unf')
                    continue
            else:
                print("Since no developement, computing new similar circuit")
                FRAME, START_FRAME, continue_loop, selected_engine, engine = process_circuits(
                    start_time,
                    total_time,
                    FRAME,
                    end_time,
                    engine_sequence,
                    count,
                    s,
                    FLAGS,
                    CURR_FRAME,
                )
                if not continue_loop:
                    break
                continue
            FRAME, START_FRAME, continue_loop, selected_engine, engine = process_circuits(
                start_time,
                total_time,
                FRAME,
                end_time,
                engine_sequence,
                count,
                s,
                FLAGS,
                CURR_FRAME,
            )
            if not continue_loop:
                break
            first_simcheck_done = True
    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        signal.alarm(0)  # Disable the alarm

def run_v_mode(args):
    with open('data/model.pkl', 'rb') as pkl_file:
        model_dict = pickle.load(pkl_file)
    feature_cols = ['Var', 'Cla','Conf','Learn']
    global TIMELIMIT, total_time, FRAME, start_time, end_time
    # circuit = args.input_circuit
    T = TIMELIMIT
    mod_time = T
    first_iter = True
    START_FRAME = 0
    CURR_FRAME = 0
    count = 1
    first_simcheck_done = False
    selected_engine = None
    print(
        f"\nTotal execution time: {total_time} seconds (Start to End of the Framework)\n"
    )
    signal.signal(
        signal.SIGALRM, terminate_process
    )  # Set the signal handler for SIGALRM
    signal.alarm(total_time)  # Schedule an alarm for the total execution time
    try:
        while time.time() - start_time <= total_time:
            s = time.time()
            print(
                "============================================================================================================================================================================================================\n"
            )
            if (
                (CURR_FRAME != START_FRAME and START_FRAME != -1)
                or first_iter
                or START_FRAME != -1
            ):
                first_iter = False
                CURR_FRAME = START_FRAME
                FLAGS = f"-S {START_FRAME} -T {TIMELIMIT} -F 0 -v"
                if first_simcheck_done:
                    last_depth_data = extract_data()
                    time_wasted = round(mod_time - float(last_depth_data[7]),2)
                    print(f'Time wasted in previous iteration is : {time_wasted} sec')
                    if time_wasted <= (args.p * mod_time):
                        '''Time prediction by model'''
                        last_depth_data = [int(value) for value in last_depth_data[1:5]]
                        predictor_data = pd.DataFrame(data = np.asarray(last_depth_data).reshape(1,-1), columns = feature_cols)
                        pred_time = model_dict[engine].predict(predictor_data)[0]
                        del_T = round(pred_time[0],2)
                        print(f'\nPredicted del_T is : {del_T} sec')
                        mod_time = round(float((mod_time + del_T)/2),2)
                        FLAGS = f"-S {START_FRAME} -T {mod_time} -F 0 -v"
                    # print(os.getcwd())
                    # os.chdir('../')
                        print(f"\nStarting at DEPTH ({START_FRAME}) : \n")
                        depth_reached = run_engine(
                            selected_engine, args.input_circuit, FLAGS
                        )
                        if depth_reached is None:
                            print(
                                f"Running {selected_engine} on {args.input_circuit.split('/')[-1]} for {mod_time} second, Depth reached : {FRAME}\n"
                            )
                            print(f"{D}\n")
                            START_FRAME = -1
                        # os.chdir('exp_unf')
                            first_simcheck_done = True
                            continue
                        if "+" in depth_reached:
                            FRAME = int(
                            depth_reached.split(":")[0].strip().split("+")[0].strip()
                            )
                        else:
                            for part in depth_reached.split("."):
                                if "F =" in part:
                                    FRAME = int(part.split("=")[1].strip())
                        START_FRAME = FRAME
                        if CURR_FRAME == START_FRAME:
                            START_FRAME = -1
                        print(
                            f"Running {selected_engine} on {args.input_circuit.split('/')[-1]} for {mod_time} second, Depth reached : {FRAME}\n"
                        )
                        print(f"{D}\n")
                        first_simcheck_done = True
                    # os.chdir('exp_unf')
                        continue
            else:
                print("Since no developement, computing new similar circuit")
                FRAME, START_FRAME, continue_loop, selected_engine, engine = process_circuits(
                    start_time,
                    total_time,
                    FRAME,
                    end_time,
                    engine_sequence,
                    count,
                    s,
                    FLAGS,
                    CURR_FRAME,
                )
                if not continue_loop:
                    break
                continue
            FRAME, START_FRAME, continue_loop, selected_engine, engine = process_circuits(
                start_time,
                total_time,
                FRAME,
                end_time,
                engine_sequence,
                count,
                s,
                FLAGS,
                CURR_FRAME,
            )
              
            if not continue_loop:
                break
            first_simcheck_done = True
    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        signal.alarm(0)  # Disable the alarm


def valid_range(value):
    damping_factor = float(value)
    if damping_factor < 0 or damping_factor > 1:
        raise argparse.ArgumentTypeError(f"Value must be between 0 and 1. Provided value : {value}")
    return damping_factor
def main():
    global args
    initial_parser = argparse.ArgumentParser(description="BMC Sequence Script", add_help=False)
    initial_parser.add_argument("-v", action="store_true", help="Run in mode --> Variable")
    initial_parser.add_argument("-f", action="store_true", help="Run in mode --> Fixed")
    initial_args, remaining_argv = initial_parser.parse_known_args()
    
    parser = argparse.ArgumentParser(description="BMC Sequence Script")
    parser.add_argument("-v", action="store_true", help="Run in mode --> Variable")
    parser.add_argument("-f", action="store_true", help="Run in mode --> Fixed")
    parser.add_argument("--input_circuit", type=str, help="Name of the input circuit")
    parser.add_argument(
        "--known_circuit_path", type=str, help="Path to the known circuits directory"
    )
    parser.add_argument("--csv_path", type=str, help="Path to the CSV directory")
    parser.add_argument("--unfold_path", type=str, help="Path to the unfold directory")
    if initial_args.v:
        parser.add_argument("--p", type=valid_range, help="Argument specific to Variable mode", default = 0.8)
    args = parser.parse_args()
    if args.f:
        run_f_mode(args)
    elif args.v:
        run_v_mode(args)
    else:
        print("Please specify a mode with -f or -v")


if __name__ == "__main__":
    main()
