import pandas as pd

# data=pd.read_csv('data/dev.tsv', sep='\t',header=None,names=['label','text'])
# for label,text in data.iterrows():
#     print(text)
#     break
#  print(data)
# with open('data/dev.tsv', 'r', encoding='utf-8') as f:
#     content = [_.strip() for _ in f.readlines()]
# for line in content:
#     parts = line.split('\t')
#     label, text = parts[0], ''.join(parts[1:])
#     print(label,text)

trans={'game':0,'fashion':1,'houseliving':2}
print(trans.get('game'))