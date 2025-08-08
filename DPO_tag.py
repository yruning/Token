# train_grpo.py
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
import numpy as np
import re
import argparse
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

#dataset = load_dataset("trl-lib/tldr", split="train")
os.environ["WANDB_MODE"] = "disabled"
# Define the reward function, which rewards completions that are close to 20 characters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ss_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

def ex_sen(text):
    split_index = [i for i, char in enumerate(text) if char in {'\n', '.'}]
    split_index.insert(0,0)
    result = []
    for i in range(len(split_index)):
        if i!= len(split_index) - 1:
            if i == 0:
                im = text[split_index[i]:min(split_index[i+1]+1, len(text))]
            else:
                im = text[split_index[i]+1:min(split_index[i+1]+1, len(text))]
        else:
            if split_index[i]!= len(text):
                im = text[min(split_index[i]+1,len(text)):]
        if im != '\n' and im!= '.' and im!='':
            result.append(im)
    s_index = []
    for i in range(len(result)):
        if (len(result[i]) >= 2 and result[i][-1] == '.' and result[i][-2].isupper()) or (len(result[i])<=3 and result[i][-1] == '.' and result[i][-2].isdigit()):
            s_index.append(i)
            if i!= len(result) - 1:
                result[i+1] = result[i] + result[i+1]
    result = [result[i] for i in range(len(result)) if i not in s_index]
    result = [item for item in result if item.strip() != "" and re.search(r"[a-zA-Z0-9]", item)]
    """
    if result[-1][0] == '#':
        result[-2]  = result[-2] + result[-1]
        result.pop()
    """
    if len(result) == 0:
        result.append(text)
    return result

def extract_number(sentence):
    result = []
    for i in range(len(sentence)):
        if (sentence[i].isdigit() and i ==0 ) or (sentence[i].isdigit() and sentence[max(i-1,0)].isdigit() == False and sentence[max(i-1,0)] != '.' and sentence[max(i-1,0)] != ',') or sentence[i] == '.' and sentence[min(len(sentence)-1,i+1)].isdigit() and sentence[max(0,i-1)].isdigit() is False:
            rs = []
            k = 0
            while sentence[i+k].isdigit() or sentence[i+k] == '.' or sentence[i+k] == ',':
                if sentence[i+k] != ',':
                    rs.append(sentence[i+k])
                if i+k<len(sentence)-1:
                    k = k+1
                else:
                    break
            
            if rs[len(rs)-1] != '.':
                result.append(''.join(rs))
            else:
                result.append(''.join(rs[:len(rs)-1]))
    return result

def L2_s(r1, r2, model):
    i_c = ex_sen(r1)
    i_m = ex_sen(r2)
    ss = np.zeros((len(i_c), len(i_m)))
    for i in range(len(i_c)):
        for j in range(len(i_m)):
            embeddings = model.encode([i_c[i], i_m[j]], convert_to_tensor=True)
            score = util.cos_sim(embeddings[0], embeddings[1])
            ss[i,j] = score.item()
    return ss


parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--r_method", type=str, required=True)
args = parser.parse_args()

data_name = args.data_name
model_name = args.model_name
r_method = args.r_method

print(device)
#model_name = 'qwen-3b'
#model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct",device_map="auto",torch_dtype=torch.bfloat16).to(device)
#tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2",device_map="auto").to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Phi-2 tokenizer's pad_token was missing. Set it to eos_token.")
special_tokens = {'additional_special_tokens': ['[Reset]']}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

#model.gradient_checkpointing_enable()
#ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct",device_map="auto",torch_dtype=torch.bfloat16)
#ref_model.resize_token_embeddings(len(tokenizer))

ref_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2",device_map="auto")
ref_model.resize_token_embeddings(len(tokenizer))

save_directory = f"./{model_name.split('/')[-1]}-with-reset-token"

train_dataset = load_dataset("json", data_files= data_name + '_data/'+data_name + "_dpo_train.json", split="train")
training_args = DPOConfig(output_dir=model_name+'_'+data_name+'_'+r_method, logging_steps=10, per_device_train_batch_size=2, num_train_epochs=10,save_steps=300,save_total_limit=1)
#trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,  # Pass the reference model here
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer, # Pass the updated tokenizer
)
trainer.train()
final_model_save_directory = f"./{model_name.split('/')[-1]}_{data_name}_{r_method}/final"
trainer.save_model(final_model_save_directory)
tokenizer.save_pretrained(final_model_save_directory)