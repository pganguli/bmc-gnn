# Folder Structure

```sh
.
├── bmc_gnn/
│   ├── cosine_similarity.py
│   ├── extract_bmc_engine.py
│   ├── extract_frame_time.py
│   ├── most_similar_circuit.py
│   └── unfold_circuit.py
├── data/
│   ├── bmc_data_circuits.txt
│   ├── bmc_data_csv.tar.xz
│   ├── bmc_data_txt.tar.xz
│   ├── circuits.tar.xz
│   ├── mab_bmc_sat_circuits_reported.txt
│   ├── mab_bmc_unsat_circuits_reported.txt
│   ├── model.pkl
│   ├── no_output_asserted_circuits_list.zip
├── main.py
├── poetry.lock
├── pyproject.toml
├── README.md
└── scripts/
    ├── bmc2_txt2csv.sh
    ├── bmc3_txt2csv.sh
    ├── create_embedding_db.sh
    ├── dump_bmc_data.mk
    ├── mlr.py
    └── rowsum_embedding.py
```
