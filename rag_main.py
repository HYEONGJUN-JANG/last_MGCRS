from tqdm import tqdm
import utils
import logging
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional
import pickle
import faiss
import torch
from datasets import Features, Sequence, Value, load_dataset, list_datasets
import datasets
# from transformers.integrations import is_ray_available
from transformers import (DPRContextEncoder, DPRContextEncoderTokenizerFast, RagRetriever, RagSequenceForGeneration, RagTokenizer, RagConfig)
# from utils_rag import (calculate_exact_match, flatten_list, is_rag_model, lmap, pickle_save,  save_json, set_extra_model_params, Seq2SeqDataset)
import pandas as pd
from collections import defaultdict
import sys
from loguru import logger


def split_documents(documents: dict) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        if text is not None:
            for passage in split_text(text):
                titles.append(title if title is not None else "")
                texts.append(passage)
    return {"title": titles, "text": texts}


def split_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i: i + n]).strip() for i in range(0, len(text), n)]


def embed(documents: dict, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast) -> dict:
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt")["input_ids"]
    embeddings = ctx_encoder(input_ids.to(device=args.device), return_dict=True).pooler_output
    return {"embeddings": embeddings.detach().cpu().numpy()}


def main():
