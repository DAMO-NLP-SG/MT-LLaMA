import torch
from torch.utils.data import Dataset
from transformers.utils import logging
import random
import os
from tqdm import tqdm
from utils import SquadV1Processor
logger = logging.get_logger(__name__)
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}
random.seed(42)


class QADataset(Dataset):
    """
    MRC QA Dataset
    """
    def __init__(self, features, tokenizer: None, data_name: None,):
        self.all_data = features
        self.data_name = data_name
        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        """
        Args:
            item: int, idx
        """
        data = self.all_data[item]
        qid, inputs, outputs = data
        return [
            torch.LongTensor(inputs['input_ids']),
            torch.LongTensor(inputs['attention_mask']),
            torch.LongTensor(outputs['input_ids']),
            torch.LongTensor(outputs['attention_mask']),
        ]


def cache_mQAexamples(args, tokenizer, dataset_name, split, lang='en'):

    num_qa = 0
    processor = SquadV1Processor()
    if split == "train" and dataset_name == "TyDiQA":
        input_data = processor.get_examples(os.path.join('./Data', 'tydiqa', 'tydiqa-goldp-v1.1-train'),
                                            "tydiqa.goldp.en.train.json")
    elif split == "dev" and dataset_name == "TyDiQA":
        input_data = processor.get_examples(os.path.join('./Data', 'tydiqa', 'tydiqa-goldp-v1.1-dev'),
                                            "tydiqa.goldp." + lang + ".dev.json")
    elif split == "train" and dataset_name == "SQuAD":
        input_data = processor.get_examples(os.path.join('./Data', 'squad'), "train-v1.1.json")
    elif split == "dev" and dataset_name == "SQuAD":
        input_data = processor.get_examples(os.path.join('./Data', 'squad'), "dev-v1.1.json")
    elif split == "dev" and dataset_name == "XQuAD":
        input_data = processor.get_examples(os.path.join('./Data', 'xquad'), "xquad." + lang + ".json")
    elif split == "dev" and dataset_name == "MLQA":
        input_data = processor.get_examples(os.path.join('./Data', 'mlqa/test'), 'test-context-' + lang + "-question-" + lang + ".json")
    features = []
    for i_t, item in enumerate(tqdm(input_data, desc="creating {} features from {}".format(split, dataset_name))):
        context = item.doc_tokens
        context_str = " ".join([x[0] for x in context])
        num_qa += 1
        qid = item.qas_id
        answers = item.answers
        query_str = item.question_text
        answer_str = answers[0]['text']
        if split != 'train':
            input_str = f'Answer the question according to the context. Question: {query_str}. Context: {context_str}. Answer:'
            inputs = tokenizer(input_str, return_tensors='pt', truncation=True, max_length=1024)
            outputs = tokenizer(answer_str, return_tensors='pt')

            features.append((qid, inputs, outputs))
        else:
            raise NotImplementedError
    logger.warning("finish creating {} features on {} questions".format(len(features), num_qa))
    return features