import json


def load_json(path_):
    with open(path_, "r", encoding="utf-8") as f:
        res = json.load(f)

    return res


def dump_json(samples, path_):

    with open(path_, "w", encoding="utf-8") as f:
        json.dump(
            samples,
            f,
            ensure_ascii=True,
            indent=2
        )


def load_jsonl(path_):
    list_samples = []
    with open(path_, "r", encoding="utf-8") as f:
        for line in f:
            list_samples.append(
                json.loads(line.strip())
            )

    return list_samples


def dump_jsonl(samples, path_):
    with open(path_, "w", encoding="utf-8") as f:
        for samp in samples:
            f.write(
                json.dumps(samp, ensure_ascii=False) + "\n"
            )

