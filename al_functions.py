import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from arabert.preprocess import ArabertPreprocessor
import pandas as pd
import numpy as np
from scipy.stats import entropy
import random, re

def transformers_batch_encoding(input_text, tokenizer):
    '''
    Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
    '''
    return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                        max_length = 32,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt'
                   )

def b_metrics(preds, labels):
    '''
    Returns the following metrics:
    - accuracy    = (TP + TN) / N
    - precision   = TP / (TP + FP)
    - recall      = TP / (TP + FN)
    '''
    preds = np.argmax(preds, axis=1).flatten()
    labels = labels.flatten()
    tp = sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])
    tn = sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])
    fp = sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])
    fn = sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])
    b_accuracy = (tp + tn) / len(labels)
    b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
    b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
    return b_accuracy, b_precision, b_recall

def create_model(model_name, num_labels):
    # Load the AutoModelForSequenceClassification model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels = num_labels,
        output_attentions = False,
        output_hidden_states = False,
    )
    # Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
    optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5, eps = 1e-08)
    return model, optimizer

def training(model, optimizer, train_dataloader, device):
    model.train()
    # Tracking variables
    tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        # Forward pass
        train_output = model(b_input_ids, token_type_ids = None, attention_mask = b_input_mask, labels = b_labels)
        # Backward pass
        train_output.loss.backward()
        optimizer.step()
        # Update tracking variables
        tr_loss += train_output.loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

def eval_active_learning(model, logits, pool_dataloader, device):
    model.eval()
    # Tracking variables 
    val_accuracy, val_precision, val_recall = [], [], []
    labels = []
    
    for batch in pool_dataloader:
        batch = tuple(t.to(device) for t in batch)
        _, _, b_labels = batch
        labels.append(b_labels)
    
    b_labels = torch.cat(tuple(b_label for b_label in labels))
    label_ids = b_labels.to('cpu').numpy()
    b_accuracy, b_precision, b_recall = b_metrics(logits, label_ids)
    val_accuracy.append(b_accuracy)
    # Update precision only when (tp + fp) !=0; replace nan with 0.0
    if b_precision != 'nan':
        val_precision.append(b_precision)
    else:
        val_precision.append(0.0)
    # Update recall only when (tp + fn) !=0; replace nan with 0.0
    if b_recall != 'nan':
        val_recall.append(b_recall)
    else:
        val_recall.append(0.0)
    return val_accuracy, val_precision, val_recall

def return_logits(model, pool_dataloader, device):
    model.eval()
    # Tracking variables 
    all_logits = []
    for batch in pool_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, _ = batch
        with torch.no_grad():
          # Forward pass
          eval_output = model(b_input_ids, 
                              token_type_ids = None, 
                              attention_mask = b_input_mask)
        logits = eval_output.logits.detach().cpu().numpy()
        for logit in logits:
            all_logits.append(logit)
    return all_logits

def preprocessing(text, label, model_name):
    arabert_prep = ArabertPreprocessor(model_name=model_name)
    text = text.apply(arabert_prep.preprocess)
    
    text, labels = text.values, label.values

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoding_dicts = [transformers_batch_encoding(sample, tokenizer) for sample in text]
    input_ids = [encoding_dict['input_ids'] for encoding_dict in encoding_dicts]
    attention_masks = [encoding_dict['attention_mask'] for encoding_dict in encoding_dicts]

    return torch.cat(input_ids,dim = 0), torch.cat(attention_masks,dim=0), torch.tensor(labels)

def calculate_entropy(logits):
    probas = torch.nn.Softmax(dim=1)(torch.from_numpy(logits))
    samples_entropy = entropy(probas.transpose(0, 1).cpu())
    samples_entropy = torch.from_numpy(samples_entropy)
    return samples_entropy

def data_prep(pool, train, model_name, batch_size):
    pool_input_ids, pool_attention_masks, pool_labels = preprocessing(pool.text, pool.label, model_name)
    train_input_ids, train_attention_masks, train_labels = preprocessing(train.text, train.label, model_name)

    pool_set = TensorDataset(pool_input_ids, pool_attention_masks, pool_labels)
    train_set = TensorDataset(train_input_ids, train_attention_masks, train_labels)

    pool_dataloader = DataLoader(pool_set, sampler=SequentialSampler(pool_set), batch_size=batch_size)
    train_dataloader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=batch_size)

    return pool_dataloader, train_dataloader
