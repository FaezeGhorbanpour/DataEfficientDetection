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
from sklearn.model_selection import train_test_split


_CITATION = """\
"""

_DESCRIPTION = """\
"""

_DATA_URL = "https://raw.githubusercontent.com/codogogo/xhate/refs/heads/main/test/"


class ParallelData(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=f'{lang}-{source}', version=datasets.Version("1.0.0"))
        for lang in ['en', 'tr', 'de', 'hr', 'ru', 'sq']
        for source in ['Gao', 'Trac', 'Wul']
    ]

    def _info(self):

        features = datasets.Features(
            {
                'id': datasets.Value('string'),
                "text": datasets.Value("string"),
                "label": datasets.features.ClassLabel(names=['0', '1']),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage="https://github.com/codogogo/xhate/tree/main",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        language = self.config.name.split('-')[0]
        source = self.config.name.split('-')[1]
        path = {}
        if language == 'en':
            path['test'] = _DATA_URL + f"{language}/XHate999-{language.upper()}-{source}-test.txt"
        elif language == 'hr' and source == 'Trac':
            path['test'] = _DATA_URL + f"{language}/XHate999-{language.upper()}-{source.upper()}.txt"
        else:
            path['test'] = _DATA_URL + f"{language}/XHate999-{language.upper()}-{source}.txt"

        dl_dir = dl_manager.download_and_extract(path)
        test_file_path = dl_dir['test']

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": test_file_path,
                    'split': 'test',
                }
            )
        ]

    def _generate_examples(self, filepath, split):
        idx = 0  # Start index from 0

        with open(filepath, encoding="utf-8", errors="replace") as csv_file:
            csv_reader = csv.reader(
                csv_file,
                quotechar='"',
                delimiter="\t",  # Assuming tab-separated values
                quoting=csv.QUOTE_ALL,
                skipinitialspace=True
            )

            # Skip header if present
            next(csv_reader, None)

            for row in csv_reader:
                if len(row) < 2:
                    print(f"Skipping invalid row: {row}")  # Debug statement
                    continue

                text, label = row[:2]  # Take the first two columns
                yield idx, {
                    'id': str(idx),
                    'text': text,
                    'label': label,
                }
                idx += 1

