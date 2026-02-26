import json

import evaluate

import sys
sys.path.append("./")


def calc_acc(list_preds, list_targets):
    num_corrects = 0
    for pred, target in zip(list_preds, list_targets):
        # print(pred, target)
        if pred == target or target in pred:
        # or target[0] in pred[: 8]:
            num_corrects += 1
    acc = num_corrects / len(list_preds)
    print(f"Accuracy: {acc:.2%} (Correct: {num_corrects}/{len(list_preds)})")
    return acc

def calc_f1_em(list_preds, list_targets):
    """
    计算 F1 和 EM 分数，常用于 QA 任务
    """
    def normalize(text):
        # 简单归一化
        return " ".join(text.lower().strip().split())

    f1_total = 0.0
    em_total = 0.0
    for pred, target in zip(list_preds, list_targets):
        pred_norm = normalize(pred)
        target_norm = normalize(target)

        # EM
        em_total += int(pred_norm == target_norm)

        # F1
        pred_tokens = pred_norm.split()
        target_tokens = target_norm.split()
        common = set(pred_tokens) & set(target_tokens)
        num_same = len(common)
        if num_same == 0:
            f1 = 0.0
        else:
            precision = num_same / len(pred_tokens)
            recall = num_same / len(target_tokens)
            f1 = 2 * precision * recall / (precision + recall)
        f1_total += f1

    em = em_total / len(list_preds)
    f1 = f1_total / len(list_preds)

    print(f"EM: {em:.2%}, F1: {f1:.2%}")
    return f1, em

def calc_mt_scores(list_preds, list_targets, metric_name="nist_mt"):
    scorer = evaluate.load(metric_name)

    results = scorer.compute(predictions=list_preds, references=list_targets)
    print("results: ", results)

    if metric_name == "rouge":
        score = results.get("rougeL")
    else:
        score = results.get(metric_name)

    return score


def calc_scores(pred_file_path):

    list_preds, list_targets = read_prediction_file(pred_file_path)

    # acc
    acc = calc_acc(list_preds, list_targets)
    print("acc: ", acc)

    # BLEU:
    # bleu_score = calc_mt_scores(list_preds, list_targets, metric_name="bleu")
    # print("bleu_score: ", bleu_score)

    # nist_mt:
    # nist_score = calc_mt_scores(list_preds, list_targets, metric_name="nist_mt")

    # meteor:
    # meteor_score = calc_mt_scores(list_preds, list_targets, metric_name="meteor")

    # rouge:
    # rougeL_score = calc_mt_scores(list_preds, list_targets, metric_name="rouge")

    # rouge:
    # CIDEr_score = calc_mt_scores(list_preds, list_targets, metric_name="CIDEr")

    return {
        "acc": acc,
        # "bleu": bleu_score,
        # "nist": nist_score,
        # "meteor": meteor_score,
        # "rougeL": rougeL_score,
        # "CIDEr": CIDEr_score,
    }


def read_prediction_file(pred_file_path):
    list_samples = []
    with open(pred_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            line = json.loads(line)
            list_samples.append(line)

    list_preds = []
    list_targets = []
    for samp in list_samples:
        pred = samp["pred"]
        target = samp["target"]
        list_preds.append(pred)
        list_targets.append(target)

    return list_preds, list_targets


if __name__ == "__main__":

    # source /etc/network_turbo

    # pred_file_path = "experiments/llama2_7b_e2e_1/test_predictions_bak1.json"
    # pred_file_path = "experiments/llama2_7b_e2e_2/test_predictions_2.json"
    # pred_file_path = "resources/Llama-2-7b-hf/test_predictions.json"
    pred_file_path = "experiments/IAAA/llama2_7b_rte_0/test_predictions.json"
    pred_file_path = "experiments/IAAA/llama2_7b_rte_3/test_predictions.json"
    score = calc_scores(pred_file_path)
    print("score: ", score)

