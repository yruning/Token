import json
import pandas as pd
from tqdm import tqdm
data_name = 'HH'
model_name = 'phi-2'

e = pd.read_excel(data_name+'_data/'+ data_name + '_'+model_name+'_basic.xls')
r = e.values

result = []
keyword = 'Human:'
for i in tqdm(range(r.shape[0])):

    if isinstance(r[i][3], str):
        index = r[i][3].find(keyword)
    else:
        index = -1
    response =  r[i][3][:index] if index != -1 else r[i][3]
    result.append({
        "question": r[i][0],
        "answer": r[i][1],
        "label": r[i][2],
        "response":response,  
        
    })
    
with open(data_name +'_data/'+data_name+'_'+model_name + '.json', "w") as f:
    json.dump(result, f, indent=2)