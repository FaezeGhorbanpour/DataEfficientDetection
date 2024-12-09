import datasets


path_name = 'parallel_data'
type_name = 'ru-Gao'
n = datasets.load_dataset(path_name, type_name, split='test', trust_remote_code=True)
print(n)