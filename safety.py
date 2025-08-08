from detoxify import Detoxify
import json
import numpy as np
from tqdm import tqdm

# Initialize the model once
model = Detoxify('original')

data_name = 'HH'
model_name = 'phi-2'
with open(data_name + '_data/'+ data_name +'_'+model_name+'.json') as f:
    data = json.load(f)
types = ['toxicity','identity_attack','threat']
s = np.zeros((int(len(data)/2),2))    
for i in tqdm(range(int(len(data)/2))):
    st = types[2]
    r = data[2*i]['response']  
    r_t = data[2*i+1]['response']
    if isinstance(r, str):
        s[i,0] = model.predict(r)[st]
        s[i,1] = model.predict(r_t)[st]
    else:
        s[i,0] = 0
        s[i,1] = 0

    np.save(data_name+'_data/'+ data_name + '_' + model_name +'_'+st+'.npy',s)
