def extract_bmc_engine(engine):
    bmc_engines = {
        "bmc2": "bmc2",
        "bmc3": "bmc3",
        "bmc3g": "bmc3 -g",
        "bmc3r": "bmc3 -r",
        "bmc3u": "bmc3 -u",
        "bmc3s": "bmc3 -s",
        "bmc3j": "bmc3J",
    }
    if engine in bmc_engines:
        selected_engine = bmc_engines[engine]
        return selected_engine
    else:
        return None
