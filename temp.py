from collections import Counter

import datasets


path_name = 'baseline_data'
type_name = 'ous19_fr'
n = datasets.load_dataset(path_name, type_name, trust_remote_code=True)
print(n)


print(Counter(n['hate_day']['label']))
