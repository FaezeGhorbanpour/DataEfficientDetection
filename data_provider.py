from datasets import load_dataset, Dataset
from datasets import DatasetDict, concatenate_datasets

class DataProvider:

    def load_datasets(self, dataset_names, languages=None, max_samples=None, cache_dir=''):
        """
        Load datasets from Hugging Face with optional filtering by language and sample limit.
        Args:
            dataset_names (list[str]): List of dataset names to load.
            languages (list[str]): List of languages to include (e.g., ["en", "fr"]).
            max_samples (int): Maximum number of samples to load per dataset.
        Returns:
            list[dict]: List of loaded datasets with metadata.
        """
        datasets = []
        for i, dataset_name in enumerate(dataset_names):
            try:
                if '/' in dataset_name:
                    dataset_name_parts = dataset_name.split('/')
                    data = load_dataset(dataset_name_parts[0], dataset_name_parts[1], cache_dir=cache_dir)
                else:
                    data = load_dataset(dataset_name, cache_dir=cache_dir)
            except:
                data = load_dataset('baseline_data', dataset_name, cache_dir=cache_dir)

            for split in data.keys():
                ds = data[split]
                # if languages:
                #     ds = ds.filter(lambda example: example.get("language") in languages)
                if split=='train' and max_samples:
                    ds = ds.select(range(min(len(ds), max_samples)))
            datasets.append({
                "name": dataset_name,
                "data": data,
                "language": languages[i]
            })
        return datasets

    def convert_to_dataset(self, retrieved_data):
        """
        Convert retrieved sentences into Hugging Face dataset format.
        Args:
            retrieved_data (list[dict]): Retrieved data with metadata and sentences.
        Returns:
            datasets.Dataset: Hugging Face Dataset object.
        """
        sentences = [item["metadata"]["text"] for item in retrieved_data]
        labels = [item["metadata"].get("label", None) for item in retrieved_data]
        scores = [item.get("score", None) for item in retrieved_data]
        return Dataset.from_dict({"text": sentences, "label": labels, "score": scores})

    def extract_text_and_label(self, retrieved_data):
        """
        Convert retrieved sentences into Hugging Face dataset format.
        Args:
            retrieved_data (dict[list]): Retrieved data with metadata and sentences.
        Returns:
            datasets.Dataset: Hugging Face Dataset object.
        """
        return {key:[{'text': value['metadata']['text'], 'label': value['metadata']['label']} for value in values] for key, values in retrieved_data.items() }

    def aggregate_splits(self, datasets, just_aggregate=[]):
        splits = set([split for ds in datasets for split in ds])

        if just_aggregate:
            return DatasetDict({
                split: concatenate_datasets([ds[split] for ds in datasets if split in ds])
                if split in just_aggregate else datasets[-1][split]
                for split in splits
            })
        else:
            return DatasetDict({
                split: concatenate_datasets([ds[split] for ds in datasets if split in ds])
                for split in splits
            })

    def combine_new_dataset(self, dataset, retrieved_data, repeat=1):
        # Add IDs and source information to the new dataset
        retrieved_data = retrieved_data.map(
            lambda example, idx: {"id": idx, "source": 1},
            with_indices=True
        )

        # Add source information to the original train dataset
        dataset["train"] = dataset["train"].map(lambda example: {"source": 0, 'score': float(0)})
        dataset["test"] = dataset["test"].map(lambda example: {"source": 0, 'score': float(0)})
        dataset["validation"] = dataset["validation"].map(lambda example: {"source": 0, 'score': float(0)})
        if 'hate_check' in dataset:
            dataset["hate_check"] = dataset["hate_check"].map(lambda example: {"source": 0, 'score': float(0)})
        if 'hate_day' in dataset:
            dataset["hate_day"] = dataset["hate_day"].map(lambda example: {"source": 0, 'score': float(0)})

        # Ensure retrieved data has the same features as the train dataset
        retrieved_data = retrieved_data.cast(dataset["train"].features)

        if repeat > 1:
            # Repeat the `train` dataset
            train = concatenate_datasets([dataset["train"]] * repeat)
        else:
            train = dataset["train"]

        # Combine the `train` subset with the new dataset
        combined_train = concatenate_datasets([train, retrieved_data])

        # Replace the `train` subset in the existing dataset with the combined dataset
        dataset['train'] = combined_train

        return dataset