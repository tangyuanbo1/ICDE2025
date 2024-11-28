from tqdm import tqdm
from datasets import load_dataset
import pandas as pd
dataset = load_dataset("Infinity-Instruct/0625", "default")

labal_to_use = ['数据科学与分析', '财务、金融与商业知识', '医学、药学与健康知识',
                '编程与软件开发', '数学能力', '法律与安全知识', '人文历史哲学与社会学知识']
domain = ['datascience', 'finance', 'medical',
          'coding', 'math', 'safety', 'social']
for label, domain in zip(labal_to_use, domain):
    cnt = 0
    data_inputs = []
    target = []
    label_for_inputs = []
    for i in tqdm(range(len(dataset['train']))):
        if (label in dataset['train'][i]['label']['cate_ability_zh']):
            data_inputs.append(dataset['train'][i]
                               ['conversations'][0]['value'])
            target.append(dataset['train'][i]['conversations'][1]['value'])
            label_for_inputs.append(
                set(dataset['train'][i]['label']['cate_ability_zh']) & set([label]))
    data = pd.DataFrame()
    data['label'] = [list(i) for i in label_for_inputs]
    data['input_text'] = data_inputs
    data['target_text'] = target
    data.to_parquet(f"../Infinity-Instruct-data/{domain}.parquet")
