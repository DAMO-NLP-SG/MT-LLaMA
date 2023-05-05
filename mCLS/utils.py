import torch
from transformers.utils import logging


logger = logging.get_logger(__name__)

def collate_to_max_length_llama(batch):
    """
    adapted form https://github.com/ShannonAI/mrc-for-flat-nested-ner
    pad to maximum length of this batch
    """
    batch_size = len(batch)
    max_length_enc = max(x[0].shape[1] for x in batch)

    output = []

    for field_idx in range(3):
        if field_idx != 1:
            pad_output = torch.full([batch_size, max_length_enc], 32000, dtype=batch[0][field_idx].dtype) # 32000 as pad token
        else:
            pad_output = torch.full([batch_size, max_length_enc], 0, dtype=batch[0][field_idx].dtype)

        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][max_length_enc - data.shape[1]:] = data # padding left
        output.append(pad_output)

    return output
