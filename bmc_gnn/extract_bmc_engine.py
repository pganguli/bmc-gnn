def extract_bmc_engine(engine):
    start_index = engine.find("bmc")
    end_index = engine.find(".csv")
    portion = None
    if start_index != -1 and end_index != -1:
        portion = engine[start_index:end_index]
    else:
        return None
    dictionary = {
        "bmc2": "bmc2",
        "bmc3": "bmc3",
        "bmc3g": "bmc3 -g",
        "bmc3r": "bmc3 -r",
        "bmc3u": "bmc3 -u",
        "bmc3s": "bmc3 -s",
        "bmc3j": "bmc3J",
    }
    if portion in dictionary:
        selected_engine = dictionary[portion]
        return selected_engine
    else:
        return None
