import glob
import os
import deepgate
import torch
import pickle
import warnings
warnings.filterwarnings("ignore")

def compare_files(file, circuit, result_list):
    max_cos = 0
    cos = torch.nn.CosineSimilarity(dim=0)
    with open(file, "rb") as pkl_file:
        friend_tensor = pickle.load(pkl_file)
    max_cos = cos(circuit, friend_tensor)
    result_list.append((file, max_cos))

def most_similar_circuit(circuit, level, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = deepgate.Model()
    model.load_pretrained()
    model = model.to(device)
    
    parser = deepgate.AigParser()
    graph_1 = parser.read_aiger(circuit)
    graph_1 = graph_1.to(device)

    hs_1, _ = model(graph_1)
    row_avg = torch.mean(hs_1, dim=0)
    result_list = []
    # Collect all relevant files
    files = []
    path =args.chosen_circuit_path
    for subdir in os.listdir(path):
        subdir_path = os.path.join(path, subdir)
        if os.path.isdir(subdir_path):
            files.extend(glob.glob(os.path.join(subdir_path, f"*_{level}.pkl")))

    for file in files:
        compare_files(file, row_avg, result_list)
    # Determine the most similar circuit
    max_similarity = -1
    most_similar_circuit = None

    for known_circuit, similarity in result_list:
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_circuit = known_circuit

    # Return the most similar circuit's filename without extension and split before '_'
    if most_similar_circuit:
        filename = os.path.basename(most_similar_circuit).split(".")[0]
        filename = filename.split('_')[0]
        return filename
    else:
        print("No similar circuit found.")
        return None
