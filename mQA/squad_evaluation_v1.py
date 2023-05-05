"""Official evaluation script for the MRQA Workshop Shared Task.
Adapted fromt the SQuAD v1.1 official evaluation script.
Usage:
    python official_eval.py dataset_file.jsonl.gz prediction_file.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import string
import re
import json
import sys
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def read_predictions(prediction_file):
    with open(prediction_file) as f:
        predictions = json.load(f)
    return predictions

def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

def compare(dataset, predictions_str, predictions_tok):
    f1 = exact_match_str = exact_match_tok = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions_str and qa['id'] not in predictions_tok:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction_str = predictions_str[qa['id']]
                prediction_tok = predictions_tok[qa['id']]
                exact_match_str_one = metric_max_over_ground_truths(
                    exact_match_score, prediction_str, ground_truths)
                exact_match_str += exact_match_str_one
                exact_match_tok_one = metric_max_over_ground_truths(
                    exact_match_score, prediction_tok, ground_truths)
                exact_match_tok += exact_match_tok_one
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction_str, ground_truths)
                if exact_match_str_one == 1 and exact_match_tok_one == 0:
                    print(qa['id'])
                    print('str answer:', prediction_str)
                    print('tok answer:', prediction_tok)
                    print('ground truth:', ground_truths)
                    print()

    exact_match = 100.0 * exact_match_str / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation for MRQA Workshop Shared Task')
    parser.add_argument('dataset_file', type=str, help='Dataset File')
    parser.add_argument('prediction_file', type=str, help='Prediction File')
    parser.add_argument('--skip-no-answer', action='store_true')
    args = parser.parse_args()

    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']
    with open(args.prediction_file+'_tok.json') as prediction_file:
        predictions_tok = json.load(prediction_file)
    with open(args.prediction_file+'_str.json') as prediction_file:
        predictions_str = json.load(prediction_file)
    compare(dataset, predictions_str, predictions_tok)
