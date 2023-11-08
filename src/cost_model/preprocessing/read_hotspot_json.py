import json


def get_hotspot_dict(file_name):
    hotspots = json.load(open(file_name, "r"))
    for k, v in hotspots.items():
        print(k, v)
        for code_name in v:
            print(code_name)


if __name__ == "__main__":
    file_name = "cBench_hotspot.json"
    get_hotspot_dict(file_name)
