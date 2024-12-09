from datasets import load_dataset, Dataset


class DataProvider:

    def load_datasets(self, dataset_names, languages=None, max_samples=None):
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
            if '/' in dataset_name:
                data = load_dataset(dataset_name.split('/')[0], dataset_name.split('/')[1])
            else:
                data = load_dataset(dataset_name)

            for split in data.keys():
                ds = data[split]
                # if languages:
                #     ds = ds.filter(lambda example: example.get("language") in languages)
                if max_samples:
                    ds = ds.select(range(min(len(ds), max_samples)))
                datasets.append({
                    "name": dataset_name,
                    "split": split,
                    "data": ds,
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
        return Dataset.from_dict({"text": sentences, "label": labels})
