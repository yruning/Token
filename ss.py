import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ss_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
data_name = 'HH'
model_name = 'qwen-3b-tag'
with open(data_name + '_data/'+ data_name +'_'+model_name+'.json') as f:
    data = json.load(f)
    
s = np.zeros(int(len(data)/2))    
for i in tqdm(range(int(len(data)/2))):
    r = data[2*i]['response']  
    r_t = data[2*i+1]['response']
    embeddings = ss_model.encode([r, r_t], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1])
    s[i] = score
    np.save(data_name+'_data/'+ data_name + '_' + model_name +'_ss.npy',s)
    