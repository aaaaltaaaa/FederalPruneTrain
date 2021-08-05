from datasets import load_dataset
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import pytreebank

def tokenize(sentences, tokenizer):
    input_ids, input_masks, input_segments = [], [], []
    for sentence in tqdm(sentences):
        inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=128, pad_to_max_length=True,
                                           return_attention_mask=True, return_token_type_ids=True,return_tensors='pt',)
        input_ids.append(inputs['input_ids'])
        input_masks.append(inputs['attention_mask'])
        input_segments.append(inputs['token_type_ids'])


class SST(Dataset):
    def __init__(self, split,max_length=128):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', mirror='tuna',
                                                             do_lower_case=True, add_special_tokens=True,
                                                             pad_to_max_length=True,
                                                             truncation=True, padding='max_length',
                                                             max_length=max_length)
        dataset = pytreebank.load_sst("/home/wnpan/feddf/data/sst/")
        data = dataset[split]
        #data = dataset['ptb_tree']
        self.text = [tree.to_lines()[0] for tree in data]
        self.targets = [tree.label for tree in data]
        self.max_length = max_length

    def tokenize(self, sentence):
        inputs = self.tokenizer.encode_plus(sentence, add_special_tokens=True, return_tensors='pt',
                                            max_length=self.max_length, pad_to_max_length=True)
        return (inputs['input_ids'][0], inputs['attention_mask'][0])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        input = self.tokenize(self.text[index])
        target = self.targets[index]
        return input, target



import torch
from pcode.models.bert import DistilBertCLS
from transformers import DistilBertConfig

dataset =SST('train')
#
# data_loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=64,
#         shuffle=True,
#         num_workers=1,
#         pin_memory=True,
#         drop_last=False,
# )
# bert_config = DistilBertConfig(num_labels=4)
#
# model = DistilBertCLS(bert_config,False).cuda()
# for data_batch,target in data_loader:
#     for i in range(len(data_batch)):
#         data_batch[i] = data_batch[i].cuda()
#     output = model(data_batch)

from torchvision.models import resnet50

model = resnet50()



