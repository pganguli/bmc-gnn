import argparse
import os
import re
import signal
import subprocess
import time

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
        # print(line)
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
    # os.chdir('circuits')
    for file in os.listdir(args.known_circuit_path):
        if time.time() - start_time > total_time:
            break  # Exit the loop if total time has exceeded
        if file.endswith(".aig") and cir_name not in file:
            unfold_circuit(f"{args.known_circuit_path}/{file}", FRAME, args.unfold_path)
    # os.chdir('../')
    cir = unfold_circuit(args.input_circuit, FRAME, args.unfold_path)
    # print(cir)
    req_cir = most_similar_circuit(cir, FRAME, args.unfold_path)
    print(req_cir)
    # os.chdir(f'../csv/{req_cir}')
    ret = extract_frame_time(FRAME, f"{args.csv_path}/{req_cir}")
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
        # os.chdir("../../")
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
            return FRAME, -1, False, None

        if "+" in depth_reached:
            FRAME = int(depth_reached.split(":")[0].strip().split("+")[0].strip())
            print(f"FINAL FRAME REACHED IS {D}\n")
        else:
            for part in depth_reached.split("."):
                if "F =" in part:
                    FRAME = int(part.split("=")[1].strip())
                    print(f"FINAL FRAME REACHED IS {FRAME}\n")
        return FRAME, -1, False, "bmc3 -g"

    selected_engine = extract_bmc_engine(engine)
    print(
        f"DEPTH ({FRAME}) : DeepGate2 execution + BMC engine selection : {round(time.time() - s, 2)} secs\n"
    )
    engine_sequence[count] = selected_engine
    count += 1
    print(
        f"Outcome at DEPTH ({FRAME}) : Most similar circuit: {req_cir}.aig, Best BMC engine for {req_cir} at Depth {FRAME} : {selected_engine}\n"
    )
    # os.chdir("../../")
    depth_reached = run_engine(selected_engine, args.input_circuit, FLAGS)
    if depth_reached is None:
        print(
            f"Running {selected_engine} on {args.input_circuit.split('/')[-1]} for {TIMELIMIT} second, Depth reached : {FRAME}\n"
        )
        print(f"\n{D}\n")
        # os.chdir("exp_unf")
        return FRAME, -1, True, selected_engine

    if "+" in depth_reached:
        FRAME = int(depth_reached.split(":")[0].strip().split("+")[0].strip())
    else:
        for part in depth_reached.split("."):
            if "F =" in part:
                FRAME = int(part.split("=")[1].strip())
    print(
        f"Running {selected_engine} on {args.input_circuit.split('/')[-1]} for {TIMELIMIT} second, Depth reached : {FRAME}\n"
    )
    print(f"\n{D}\n")
    # os.chdir("exp_unf")
    if CURR_FRAME == FRAME:
        return FRAME, -1, True, selected_engine
    return FRAME, FRAME, True, selected_engine


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
                    mod_time = 2 * mod_time
                    FLAGS = f"-S {START_FRAME} -T {mod_time} -F 0 -v"
                    # print(os.getcwd())
                    # os.chdir('../')
                    print(f"Starting at DEPTH ({START_FRAME}) : \n")
                    depth_reached = run_engine(
                        selected_engine, args.input_circuit, FLAGS
                    )
                    if depth_reached is None:
                        print(
                            f"Running {selected_engine} on {args.input_circuit.split('/')[-1]} for {mod_time} second, Depth reached : {FRAME}\n"
                        )
                        print(f"\n{D}\n")
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
                    print(f"\n{D}\n")
                    first_simcheck_done = True
                    # os.chdir('exp_unf')
                    continue
            else:
                print("Since no developement, computing new similar circuit")
                FRAME, START_FRAME, continue_loop, selected_engine = process_circuits(
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
            FRAME, START_FRAME, continue_loop, selected_engine = process_circuits(
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


def main():
    global args
    parser = argparse.ArgumentParser(description="BMC Sequence Script")
    parser.add_argument("-f", action="store_true", help="Run in mode --> Fixed")
    parser.add_argument("--input_circuit", type=str, help="Name of the input circuit")
    parser.add_argument(
        "--known_circuit_path", type=str, help="Path to the known circuits directory"
    )
    parser.add_argument("--csv_path", type=str, help="Path to the CSV directory")
    parser.add_argument("--unfold_path", type=str, help="Path to the unfold directory")
    args = parser.parse_args()
    if args.f:
        run_f_mode(args)
    else:
        print("Please specify a mode with -f")


if __name__ == "__main__":
    main()
