from datasets import load_dataset
import numpy as np
import json
from tqdm import tqdm

def save_to_json(filename, trajectory):
    """Save the trajectory to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(trajectory, f, indent=4)

def extract_prefix(text, keyword="Assistant:"):
    # Step 1: Find and extract text after the keyword
    index = text.find(keyword)
    if index != -1:
        after_text = text[index + len(keyword):].strip()
    else:
        after_text = text

    # Step 2: Split by '.' into sentences (remove empty/extra spaces)
    sentences = [s.strip() for s in after_text.split('.') if s.strip()]

    # Step 3: Take first sentence
    if not sentences:
        return ""

    summary = sentences[0]

    # Step 4: Check word count, append second sentence if < 20 words
    if len(summary.split()) < 20 and len(sentences) > 1:
        summary += '. ' + sentences[1]

    return summary.strip() + '.'

"""
ds = load_dataset("Anthropic/hh-rlhf")
data = ds['train']
ds = load_dataset("HuggingFaceH4/helpful-anthropic-raw")
data2 = ds['train']
keyword = 'Assistant:'
c = 0
s_index = []
for i in range(len(data)):
    n = np.max((data[i]['chosen'].count(keyword), data[i]['rejected'].count(keyword)))
    if n == 1:
        s_index.append(i)
        c+= 1
        
print(c)
ds = load_dataset("HuggingFaceH4/helpful-anthropic-raw")
data2 = ds['train']

n_sample = 2000

r = []
for i in tqdm(range(n_sample)):
    q_harm = data[s_index[i]]['chosen']
    index = q_harm.find(keyword)
    q_harm =  q_harm[:index] if index != -1 else q_harm
    rc_harm = data[s_index[i]]['chosen'][index+10:]
    rr_harm = data[s_index[i]]['rejected'][index+10:]
    q_help = data2[i]['instruction']
    harm_prefix = extract_prefix(data[s_index[i]]['rejected'])
    help_prefix = extract_prefix(data2[i]['demonstration'])
    r.append({
        "prompt":q_harm,
        "chosen":harm_prefix + '[Reset]' + rc_harm,
        "rejected":rr_harm,
        "label": "harm"
        
    })
    r.append({
        "prompt":q_help,
        "chosen": data2[i]['demonstration'],
        "rejected":help_prefix + '[Reset]',
        "label": "help"
    })
    
    save_to_json( 'HH_dpo_train.json', r)
"""

ds = load_dataset("Anthropic/hh-rlhf")
data = ds['test']
keyword = 'Assistant:'
s_index = []
for i in range(len(data)):
    n = np.max((data[i]['chosen'].count(keyword), data[i]['rejected'].count(keyword)))
    if n == 1:
        s_index.append(i)
        
n_sample = 200
ds = load_dataset("HuggingFaceH4/helpful-anthropic-raw")
data2 = ds['train']
r = []
for i in tqdm(range(n_sample)):
    
    q_harm = data[s_index[i]]['chosen']
    index = q_harm.find(keyword)
    q_harm =  q_harm[:index] if index != -1 else q_harm
    rc_harm = data[s_index[i]]['chosen'][index+10:]
    rr_harm = data[s_index[i]]['rejected'][index+10:]
    harm_prefix = extract_prefix(data[s_index[i]]['rejected'])
    
    r.append({
        "prompt":q_harm,
        "chosen": rc_harm,
        "rejected":rr_harm,
        "label": "base"
        
    })
    r.append({
        "prompt":q_harm + '[Reset]',
        "chosen": rc_harm,
        "rejected":rr_harm,
        "label": "tag"
        
    })
    
    q_help = data2[i+2000]['instruction']
    r.append({
        "prompt":q_help,
        "chosen": data2[i+2000]['demonstration'],
        "rejected":data2[i+2000]['demonstration'],
        "label": "base"
        
    })
    r.append({
        "prompt":q_help + '[Reset]',
        "chosen": data2[i+2000]['demonstration'],
        "rejected":data2[i+2000]['demonstration'],
        "label": "tag"
        
    })
    save_to_json( 'HH_dpo_test.json', r)