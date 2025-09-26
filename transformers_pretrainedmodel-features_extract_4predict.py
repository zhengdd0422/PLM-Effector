import torch
import numpy as np
import os
import gc
from transformers import AutoTokenizer, AutoModel
from transformers import T5Tokenizer, T5EncoderModel
from utils import read_preprocess_4pretrained_4terminal_4predict, read_preprocess_4pretrained_4predict, batch_extract_features
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_type", default='ProtBert', choices=['ProtBert', 'ProtT5', 'esm1', 'esm2_t33'], type=str)
parser.add_argument("--usefile_id", type=str, default='1', help="the path of the data sets")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(__file__)
data_path = os.path.join(BASE_DIR, "tmp")
# parameters
model_paths = {
    'ProtBert': os.path.join(BASE_DIR, "transformers_pretrained", "prot_bert"),
    'ProtT5': os.path.join(BASE_DIR, "transformers_pretrained", "prot_t5_xl_uniref50"),
    'esm1': os.path.join(BASE_DIR, "transformers_pretrained", "esm1b_t33_650M_UR50S"),
    'esm2_t33': os.path.join(BASE_DIR, "transformers_pretrained", "esm2_t33_650M_UR50D"),
}

model_maxlength = {
            'ProtBert': 512,
            'ProtT5': 512,
            'esm1': 1024,
            'esm2_t33': 1024,
        }
real_sequences = {
            'ProtBert': 510,
            'ProtT5': 511,
            'esm1': 1022,
            'esm2_t33': 1022,
        }
if args.pretrained_type == "ProtBert":
    tokenizer = AutoTokenizer.from_pretrained(model_paths[args.pretrained_type], do_lower_case=False)
elif args.pretrained_type == "ProtT5":
    tokenizer = T5Tokenizer.from_pretrained(model_paths[args.pretrained_type], do_lower_case=False)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_paths[args.pretrained_type])
# model:
if args.pretrained_type == "ProtT5":
    model = T5EncoderModel.from_pretrained(model_paths[args.pretrained_type])
else:
    model = AutoModel.from_pretrained(model_paths[args.pretrained_type])
# model = torch.nn.DataParallel(model)
model = model.to(device)
model.eval()
# Nterminal, Cterminal
for terminal in ["Nterminal", "Cterminal"]:
    if terminal == "Cterminal":
        seq_ids, sequences = read_preprocess_4pretrained_4terminal_4predict(data_path, args.usefile_id + ".fasta", args.pretrained_type, terminal=terminal, maxlen=real_sequences[args.pretrained_type])
        tokenizer.padding_side = "left"
    else:
        seq_ids, sequences = read_preprocess_4pretrained_4predict(data_path, args.usefile_id + ".fasta", args.pretrained_type)
        tokenizer.padding_side = "right"
    with torch.no_grad():
        embeddings_data, attention_masks_data = batch_extract_features(sequences, args.pretrained_type, model, tokenizer, device, max_length=model_maxlength[args.pretrained_type], batch_size=5)
    save_file = os.path.join(data_path, f"{args.usefile_id}_{args.pretrained_type}_{terminal}.npz")
    np.savez(save_file, embedding=embeddings_data, attention_masks=attention_masks_data, seq_ids=np.array(seq_ids))
    del embeddings_data, attention_masks_data, seq_ids, sequences
    gc.collect()
    torch.cuda.empty_cache()
