import datasets


path_name = 'baseline_data'
type_name = 'san20_it-2000-rs10'
n = datasets.load_dataset(path_name, type_name, trust_remote_code=True)
print(n)