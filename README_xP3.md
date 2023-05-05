# Multi-task LLaMA
This repository provides training and evaluation code for Multi-task LLaMA (MT-LLaMA), a [LLaMA](https://github.com/facebookresearch/llama) variant fine-tuned on multi-task natural language prompted datasets.
                                                                                                 
```bash
# Sentiment Analysis:
Human: a film made with as little wit, interest, and professionalism as artistically possible for a slummy hollywood caper flick. Based on this review, would the user recommend this product? No or Yes?
MT-LLaMA: no

# Multi-Choice Question Answering:
Human: There is most likely going to be fog around: Which is the correct answer? - (A) a marsh - (B) a tundra - (C) the plains - (D) a desert
MT-LLaMA: A

# Sentence Selection:
Human: The flame on the candle went out. This happened because... Help me pick the more plausible option: - I blew on the wick.. - I put a match to the wick.
MT-LLaMA: I blew on the wick..

# Natural Language Inference
Human: The new rights are nice enough Question: Does this imply that Everyone really likes the newest benefits ? Please response with 'Yes', 'No', or 'Maybe'.
MT-LLaMA: Yes, everyone likes the newest benefits.
```                                                                                                 
                                                                                                 
## Model Weights
Our MT-LLaMA is already on the Huggingface Model Hub. Since the application of original LLaMA weights is required, we only upload the weight delta against original LLaMA 
* Get the original LLaMA weights in the huggingface format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
* Download the weight delta from [here](https://huggingface.co/lmsys/fastchat-t5-3b-v1.0) and obtain the weights of MT-LLaMA by adding delta weights to the original ones.
```bash

python3 /mtlama/model/apply_delta.py \
    --base-model-path /path/to/llama-7b \
    --target-model-path /output/path/to/mt-llama-7b \
    --delta-path /path/to/mt-llama-7b-delta
```
                                        
## Prerequisite
```bash
# install pytorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# install flash-attention
pip install flash-attn==0.2.8

# install other dependencies
pip install -r requirement.txt
```
                                        
## Train Models
We fine-tune LLMs with human language instructions from massive tasks in [P3](https://huggingface.co/datasets/bigscience/P3) (i.e., T0 Train). Concretely, the used datasets during training and task taxonomy are listed below:
* Multi-choice QA: CommonsenseQA, Cosmos QA, DREAM, QuAIL, QuaRTz, QASC, QuaRel, SciQ, Social IQA, Wiki Hop, WiQA  
* Extractive QA: Adversarial QA, DuoRC, Quoref, ROPES  
* Close-Book QA: Hotpot QA, Wiki QA  
* Sentiment Classification: Amazon, App Reviews, IMDB, Rotten Tomatoes, Yelp  
* Topic Classification: AG News, DBPedia, TREC  
* Structure-to-Text Generation: Common Gen, Wiki Bio  
* Text Summarization: CNN Daily Mail, Gigaword, MultiNews, SamSum, XSum  
* Paraphrase Identification: MRPC, PAWS, QQP  

#### Sampling Strategy
We exactly follow the downsampling strategy of [Bigscience T0](https://openreview.net/forum?id=9Vrb9D0WI4) to get the mixture data of multiple tasks. 
Specifically, for extremely large datasets with over 500’000 examples, we regard it as having 500’000 / num_templates examples, where num_templates is the number of templates created by [Bigscience T0](https://openreview.net/forum?id=9Vrb9D0WI4) for the dataset.
We then do sampling according to the normalized size of the above datasets. 
Totally, we sample 20'000'000 examples for training.  


                                                                              
#### Hyperparameter Settings
| Effective Batch Size | Learning rate | Steps | Max length | Weight decay |
| :--- | --- | --- | --- | --- |
| 16 | 5e-6 | 300K | 2048 | 0 |                                                                              
                                                                              
                                          
## Zero-shot Evaluation
We primarily follow the protocols of [Bigscience T0](https://openreview.net/forum?id=9Vrb9D0WI4) to assess the generalization capability of our Multi-task LLaMA to: (1) _**Unseen Datasets**_ (i.e., datasets from seen tasks); (2) _**Unseen Tasks**_.
                                                     
#### Prompt Format                                                     
Extractive QA:

1. XQuAD, TyDiQA, MLQA, SQuAD
   ```angular2html
    Input: Answer the question according to the context. Question: {question}. Context: {context}. Answer:
    Output: {Answer}
   ```

Sentiment:

1. SST-2
   ```angular2html
   Input: {sentence} Based on this review, would the user recommend this product? No or Yes?
   Output: Yes / No
   ```
Multiple-Choice QA:

1. OpenbookQA
   ```angular2html
   Input: {question} Which is the correct answer? - (A) {choice1} - (B) {choice2} - (C) {choice3} - (D) {choice4}
   Output: {choice1} / {choice2} / {choice3} / {choice4}
   ```
Sentence Completion:

1. COPA
   ```angular2html
   Input: {premise} {% if question == "cause" %} This happened because... {% else %} As a consequence... Help me pick the more plausible option: - {choice1} - {choice2}
   Output: {choice1} / {choice2}
   ```
Coreference Resolution:
1. Winogrande:
   ```angular2html    
   Input: {sentence} In the previous sentence, does _ refer to {option1} or {option2}?
   Output: {option1} / {option2}
   ```
Word Sense Disambiguation:
1. WiC
   ```angular2html
   Input: Does the word "{word}" have the same meaning in these two sentences? Yes, No? {sentence1} {sentence2}
   Output: {sentence1} / {sentence2}
   ```
Natural Language Inference:

1. MNLI:
   ```angular2html
   Input: {premise} Question: Does this imply that {hypothesis}? Please response with 'Yes', 'No', or 'Maybe'.
   Output: Yes / No / Maybe
   ```
2. RTE
   ```angular2html  
   Input: Given {premise} Is it guaranteed true that "{hypothesis}"? Yes or no?
   Output: Yes / no
   ```
#### Results on _Unseen Datasets_

| Model       | XQuAD-en (F1/EM) | TyDiQA-en (F1/EM) | MLQA-en (F1/EM) | SQuAD (F1/EM) | SST-2 (Acc.) | OpenbookQA (Acc.) |
|:------------|------------------|-------------------|-----------------|---------------|--------------|-------------------|
| LLaMA-7b    | 9.5 / 2.0        | 14.3 / 2.6        | 13.4 / 3.3      | 29.4 / 11.5   | 50.5         | 32.4              |
| MT-LLaMA-7b | 42.3 / 31.1      | 38.9 / 26.9       | 45.4 / 31.5     | 85.9 / 77.6   | 92.6         | 38.2              |
#### Results on _Unseen Tasks_                                                     
| Model       | COPA (Acc.) | Winogrande (Acc.)  | WiC (Acc.) | MNLI (Acc.) | RTE (Acc.) |
|:------------|-------------|--------------------|------------|-------------|------------|
| LLaMA-7b    | 56.0        | 49.3               | 51.7       | 30.2        | 52.7       |
| MT-LLaMA-7b | 88.0        | 54.9               | 52.2       | 49.6        | 79.1       |