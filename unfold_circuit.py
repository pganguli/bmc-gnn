import subprocess
import os

def unfold_circuit(circuit_file, level):
    # Split path and file, and strip extension
    base_dir, filename = os.path.split(circuit_file)
    name, _ = os.path.splitext(filename)
    if not os.path.exists(name): 
        os.makedirs(name) 
    output_file = f"{name}_unf{level}.aig"
    output_path = f"{os.path.join(os.getcwd(), name, output_file)}"
    # Construct command string
    command_string = f"read_aiger {circuit_file}; &get; &frames -F {level} -s -b; &write {output_path}"

    # Execute command with stdout and stderr suppression
    try:
        subprocess.run(["abc", "-c", command_string], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return None

    return output_path
