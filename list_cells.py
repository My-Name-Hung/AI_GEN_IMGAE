import json
nb = json.load(open(r'c:\Users\.Freelancer\AI_GEN\kaggle_backend.ipynb', 'r', encoding='utf-8'))
lines = []
for i, c in enumerate(nb['cells']):
    src = c['source']
    if src:
        first = src[0][:80].replace('\n', ' ')
        lines.append(f"{i}: {first}")
    else:
        lines.append(f"{i}: EMPTY")
with open(r'c:\Users\.Freelancer\AI_GEN\cell_list.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
