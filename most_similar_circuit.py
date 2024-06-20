import os
import glob
import cosine_similarity as cs
from multiprocessing import Process, Manager

def compare_files(file_list, circuit, result_list):
    local_max_similarity = -1
    local_most_similar_circuit = None
    for file_path in file_list:
        try:
            similarity = cs.cosine_compare(circuit, file_path)
            similarity_value = similarity[1].item()
            if similarity_value > local_max_similarity:
                local_max_similarity = similarity_value
                local_most_similar_circuit = similarity[0]
        except Exception as e:
            print(f"Error comparing file {file_path}: {str(e)}")
    result_list.append((local_most_similar_circuit, local_max_similarity))

def most_similar_circuit(circuit, level, num_processes=8):
    cir_name = (circuit.split("/")[-1]).split("_")
    current_directory = os.getcwd()
    manager = Manager()
    result_list = manager.list()
    processes = []

    # Collect all relevant files
    files = []
    for subdir in os.listdir(current_directory):
        if cir_name[0] not in subdir:
            subdir_path = os.path.join(current_directory, subdir)
            if os.path.isdir(subdir_path):
                files.extend(glob.glob(os.path.join(subdir_path, f"*_unf{level}.aig")))
    
    # Split files into chunks for each process
    file_chunks = [files[i::num_processes] for i in range(num_processes)]
    
    # Start processes
    for chunk in file_chunks:
        p = Process(target=compare_files, args=(chunk, circuit, result_list))
        processes.append(p)
        p.start()
    
    # Join processes
    for p in processes:
        p.join()
    
    # Determine the most similar circuit
    max_similarity = -1
    most_similar_circuit = None
    for circuit_path, similarity in result_list:
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_circuit = circuit_path
    # Return the most similar circuit's filename without extension and split before '_'
    if most_similar_circuit:
        filename = os.path.basename(most_similar_circuit)
        filename_without_extension = os.path.splitext(filename)[0]
        filename_before_underscore = filename_without_extension.split('_')[0]
        return filename_before_underscore
    else:
        print("No similar circuit found.")
        return None




