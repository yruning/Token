#collect
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os
import torch
import json
import xlwt
from tqdm import tqdm
from huggingface_hub import login
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--strategy", type=str, required=True)
parser.add_argument("--model_para", type=str, required=True)
args = parser.parse_args()

data_name = args.data_name
model_name = args.model_name
strategy = args.strategy
model_parameter = args.model_para
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct").to(device)
#tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

#model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2").to(device)
#tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")


#checkpoint_dir = 'qwen-3b_HH_dpo_tag/checkpoint-20000'
checkpoint_dir = 'phi-2_HH_dpo_tag/checkpoint-20000'
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16 # Use the same dtype as during training
)


r1 = []
n_sample = 1

with open(data_name+'_data/'+data_name+'_test.json', "r") as f:
    raw_data = json.load(f)

if len(raw_data)>=200:
    raw_data = raw_data[:200]
N = len(raw_data)

for itr in tqdm(range(N)):
    q = raw_data[itr]['prompt']
    answer = raw_data[itr]['chosen']
    label = raw_data[itr]['label']
    inputs = tokenizer(q, return_tensors="pt").to(device)
    sample_answers = []
    for _ in range(n_sample):
        # Generate output
        output = model.generate(
            inputs.input_ids,
            max_new_tokens=1200,         # Maximum length of the generated response
            temperature=0.2,  # Set the temperature
            num_return_sequences=1,  # Generate one response at a time
            do_sample=True           # Enable sampling (not greedy decoding)
        )
        # Decode the output to text
        response_text = tokenizer.decode(output[0], skip_special_tokens=False)
        response_text = response_text[len(q):].strip()
        sample_answers.append(response_text)

    r1.append([q, answer, label] +sample_answers)
    #print(sample_answers[0])
    
    #print(sample_answers[0])
    n_o = 1
    file = xlwt.Workbook('encoding = utf-8')
    sheet1 = file.add_sheet('sheet1',cell_overwrite_ok = True)
    for i in range(len(r1)):
        for j in range(len(r1[i])):
            sheet1.write(i+n_o,j,str(r1[i][j]))
    file.save(data_name + '_data/' + data_name+'_'+model_name+'_'+strategy+'.xls')