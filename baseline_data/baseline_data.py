# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Hate speech dataset"""


import csv
import os
import random

import datasets
from datasets import Split, NamedSplit, NamedSplitAll
from sklearn.model_selection import train_test_split


_CITATION = """\
"""

_DESCRIPTION = """\
"""

_DATA_URL = "/mounts/data/proj/faeze/data_efficient_hate/datasets/main/1_clean/"
_HATE_CHECK_URL = "/mounts/data/proj/faeze/data_efficient_hate/datasets/hatecheck/"
_HATE_DAY_URL = '/mounts/data/proj/faeze/data_efficient_hate/datasets/hateday/'

class MySplit(Split):
    # pylint: disable=line-too-long
    """`Enum` for dataset splits.

    Datasets are typically split into different subsets to be used at various
    stages of training and evaluation.

    - `TRAIN`: the training data.
    - `VALIDATION`: the validation data. If present, this is typically used as
      evaluation data while iterating on a model (e.g. changing hyperparameters,
      model architecture, etc.).
    - `TEST`: the testing data. This is the data to report metrics on. Typically
      you do not want to use this during model iteration as you may overfit to it.
    - `ALL`: the union of all defined dataset splits.

    All splits, including compositions inherit from `datasets.SplitBase`.

    See the [guide](../load_hub#splits) on splits for more information.

    Example:

    ```py
    >>> datasets.SplitGenerator(
    ...     name=datasets.Split.TRAIN,
    ...     gen_kwargs={"split_key": "train", "files": dl_manager.download_and extract(url)},
    ... ),
    ... datasets.SplitGenerator(
    ...     name=datasets.Split.VALIDATION,
    ...     gen_kwargs={"split_key": "validation", "files": dl_manager.download_and extract(url)},
    ... ),
    ... datasets.SplitGenerator(
    ...     name=datasets.Split.TEST,
    ...     gen_kwargs={"split_key": "test", "files": dl_manager.download_and extract(url)},
    ... )
    ```
    """

    # pylint: enable=line-too-long
    TRAIN = NamedSplit("train")
    TEST = NamedSplit("test")
    VALIDATION = NamedSplit("validation")
    HATE_CHECK = NamedSplit("hate_check")
    HATE_DAY = NamedSplit("hate_day")
    ALL = NamedSplitAll()


class BaselineData(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=f'{source}-{shot}-{batch}'.replace('--', '-')[:-1], version=datasets.Version("1.0.0"))
        # for lang in ['en', 'es', 'pt', 'hi', 'ar', 'fr', 'it']
        for source in ['bas19_es', 'dyn21_en', 'for19_pt', 'fou18_en',
                       'has19_hi', 'has20_hi', 'has21_hi', 'ken20_en',
                       'ous19_ar', 'ous19_fr', 'san20_it', 'gahd24_de',
                       'xdomain_en', 'xdomain_tr', 'xplain_en', 'implicit_en'
                       ]
        for shot in ['', '10-', '20-', '30-', '40-', '50-', '100-', '200-', '300-', '400-', '500-', '1000-', '2000-',
                     '3000-', '4000-', '5000-', '10000-', '20000-']
        for batch in [''] + ['rs' + str(i) + '-' for i in range(1, 11)]

    ]

    def _info(self):

        features = datasets.Features(
            {
                "id": datasets.Value('string'),
                "text": datasets.Value("string"),
                "label": datasets.features.ClassLabel(names=['0', '1']),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage="https://raw.githubusercontent.com/paul-rottger/efficient-low-resource-hate-detection/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        matching_names = {
            'bas19_es':{'train':'4100'},
            'dyn21_en':{'train':'38644'},
            'for19_pt':{'train':'3170'},
            'fou18_en':{'train':'20065'},
            'has19_hi':{'train':'4165', 'test':'1318'},
            'has20_hi':{'train':'2463', 'test':'1269'},
            'has21_hi':{'train':'2094',},
            'ken20_en':{'train':'20692'},
            'ous19_ar':{'train':'2053', 'test':'1000', 'dev':'300'},
            'ous19_fr':{'train':'2014', 'test':'1500'},
            'san20_it':{'train':'5600'},
            'gahd24_de':{'train':'7701'},
            'xdomain_tr':{'train':'42659'},
            'xdomain_en':{'train':'53600'},
            'measure_en':{'train':'19526'},
            'implicit_en':{'train':'18980'},
            'xplain_en':{'train':'10999'},
            }

        source = self.config.name
        path = {}
        _, language = source.split('_')
        if '-' in source:
            source, few, batch = source.split('-')
            _, language = source.split('_')
            if language != 'en' and int(few) > 2000:
                raise Exception('Your requested train set does not exist!')
            path['train'] = _DATA_URL + f"{source}/train/train_{few}_{batch}.csv"
        else:
           path['train'] = _DATA_URL + f"{source}/train_{matching_names[source]['train']}.csv"
        path['test'] = _DATA_URL + f"{source}/test_{matching_names[source].get('test', '2000')}.csv"
        path['validation'] = _DATA_URL + f"{source}/dev_{matching_names[source].get('dev', '500')}.csv"

        dl_dir = dl_manager.download_and_extract(path)
        output = [
            datasets.SplitGenerator(
                name=MySplit.TRAIN,
                gen_kwargs={
                    "filepath": dl_dir['train'],
                    'split': 'train',
                }
            ),
            datasets.SplitGenerator(
                name=MySplit.VALIDATION,
                gen_kwargs={
                    "filepath": dl_dir['validation'],
                    'split': 'validation',
                }
            ),
            datasets.SplitGenerator(
                name=MySplit.TEST,
                gen_kwargs={
                    "filepath": dl_dir['test'],
                    'split': 'test',
                }
            )]

        if language not in ['tr', 'en']:
            path['hate_check'] = _HATE_CHECK_URL + f"hatecheck_cases_final_{language}.csv"
            dl_dir = dl_manager.download_and_extract(path)
            output += [
                datasets.SplitGenerator(
                    name=MySplit.HATE_CHECK,
                    gen_kwargs={
                        "filepath": dl_dir['hate_check'],
                        'split': 'hate_check',
                    }
                )]
        if language not in ['it']:
            path['hate_day'] = _HATE_DAY_URL + f"hateday_hf.csv"
            dl_dir = dl_manager.download_and_extract(path)
            output += [
                datasets.SplitGenerator(
                    name=MySplit.HATE_DAY,
                    gen_kwargs={
                        "filepath": dl_dir['hate_day'],
                        'split': 'hate_day',
                    }
                )]

        return output


    def _generate_examples(self, filepath, split):
        idx = -1  # Start index from 0
        source = self.config.name.split('-')[0]
        _, language = source.split('_')
        COUNTRY_LANGUAGE_MAPPING = {
            'Arabic': 'ar',
            'Spanish': 'es',
            'Portuguese': 'pt',
            'Italian': 'it',
            'Germany': 'de',
            'German': 'de',
            'India': 'hi',
            'English': 'en',
            'Indonesian': 'in',
            'Turkish': 'tr',
            'French': 'fr',
            'Kenya': 'ko',
            'Nigeria': 'ni',
            'United States': 'us',

        }
        num_non_hate = 0
        with open(filepath, encoding="utf-8", errors="replace") as csv_file:
            csv_reader = csv.reader(
                csv_file,
                quotechar='"',
                delimiter=",",
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True
            )

            next(csv_reader, None)

            for row in csv_reader:
                if len(row) < 2:
                    # print(f"Skipping invalid row: {row}")  # Debug statement
                    continue

                if split == 'hate_check':
                    idx, _, text, label = row[:4]
                elif split == 'hate_day':
                    idx, text, label, _, country = row
                    if COUNTRY_LANGUAGE_MAPPING[country] != language:
                        continue

                    label = float(label)
                    if (int(label) == 0 or int(label) == 1) and num_non_hate < 1800:
                        label = 0
                        num_non_hate += 1
                    elif int(label) == 2:
                        label = 1
                    else:
                        continue
                    # print(country, text)
                else:
                    text, label = row[:2]  # Take the first two columns
                    idx += 1

                yield idx, {
                    'id': str(idx),
                    'text': text,
                    'label': label,
                }



