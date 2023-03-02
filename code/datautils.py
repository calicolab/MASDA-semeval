import json
import torch
import string
import numpy as np
import pytorch_lightning as pl

from tqdm import tqdm
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, pipeline
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Sequence, Tuple


ALPH = tuple([i for i in string.ascii_letters])
WATCHLIST = ('user', 'url')

class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


class TextModifier(ABC):
    @abstractmethod
    def __call__(
        self, samples: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        pass


class MdaDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, any]],
        inference_mode: bool = False
    ):
        self.samples: List[Tuple[str, str, str, int,\
            Tuple[float, float]]] = list()

        for sample in samples:
            if inference_mode:
                self.samples.append((
                    sample['subsetId'],
                    sample['text'],
                    sample['other_info']['domain']
                ))
                continue

            self.samples.append((
                sample['subsetId'],
                sample['text'],
                sample['other_info']['domain'],
                sample['hard_label'],
                (sample['soft_label']['0'], sample['soft_label']['1'])
            ))      

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class MdaBatcher:
    def __init__(self, tnkzr_path: str, has_targets=True) -> None:
        """
        Args:
            tnkzr_path (str): Path to load the `Transformers` tokenizer
            to be used.
            has_targets (bool): Does the dataset have target information.
            Defaults to True.
        """
        self.has_targets = has_targets
        self.tokenizer = AutoTokenizer.from_pretrained(tnkzr_path)

    def __call__(self, batch: Sequence):
        """Use this function as the `collate_fn` mentioned earlier.
        """
        ids = torch.tensor([int(sample[0]) for sample in batch], dtype=torch.int32)
        tokens = self.tokenizer(
            [sample[1] for sample in batch], # The text
            [sample[2] for sample in batch], # The domain
            max_length=128, # As per https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#fine-tuning
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )

        if not self.has_targets:
            return ids, tokens

        targets = torch.tensor([[float(sample[3])] for sample in batch], dtype=torch.float32)
        soft_targets = torch.tensor([sample[4] for sample in batch], dtype=torch.float32)

        return ids, tokens, (targets, soft_targets)

class MdaAugmenter(TextModifier):
    def __init__(
        self,
        model_name: str,
        top_k: int = 10,
        device: int = 0
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mask_filler = pipeline(
            model=model_name,
            top_k=top_k,
            device=device
        )

    def masker(self, text: str):
        tokens = self.tokenizer.tokenize(text)
        masked = False

        # Mask a word
        chosen_indices: int = list()
        rng = np.random.default_rng(seed=44)
        while True:
            if len(chosen_indices) == len(tokens):
                break

            mask_idx = rng.integers(len(tokens))

            if (not tokens[mask_idx].startswith(ALPH, 1)) or \
                tokens[mask_idx] in WATCHLIST:
                continue

            tokens[mask_idx] = self.tokenizer.special_tokens_map['mask_token']
            break

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return self.tokenizer.decode(tokens_ids)

    def __call__(self, samples: List[str]):
        masked_samples: List[str] = list()
        aug_samples: List[Dict[str, any]] = list()
        for sample in samples:
            # Append the base sample
            aug_samples.append(sample.copy())

            masked_samples.append(self.masker(sample['text']))

        # Batch fill the masks
        filled_samples_list: List[List[Dict[str, any]]] = tqdm(self.mask_filler(
            ListDataset(masked_samples),
            batch_size=16
        ))

        for filled_samples, sample in zip(filled_samples_list, samples):
            for idx, filled_sample in enumerate(filled_samples):
                sample_copy = sample.copy()
                sample_copy.update({
                    'subsetId': str((idx+1)*int(1e7)+int(sample_copy['subsetId'])),
                    'text': filled_sample['sequence']
                })
                aug_samples.append(sample_copy)

        return aug_samples


class MdaMasker(TextModifier):
    def __init__(
        self,
        model_name: str,
        mask_perc: int = 15,
        keep_base: bool = True
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mask_perc = mask_perc
        self.keep_base = keep_base

    def masker(self, text: str):
        tokens = self.tokenizer.tokenize(text)

        # Mask a word
        chosen_indices: int = list()
        rng = np.random.default_rng(seed=44)
        while True:
            if len(chosen_indices) == len(tokens):
                break

            mask_idx = rng.integers(len(tokens))
            chosen_indices.append(mask_idx)

            if (not tokens[mask_idx].startswith(ALPH, 1)) or \
                tokens[mask_idx] in WATCHLIST:
                continue

            tokens[mask_idx] = self.tokenizer.special_tokens_map['mask_token']

            if len(chosen_indices) >= self.mask_perc / 100 * len(tokens):
                break

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return self.tokenizer.decode(tokens_ids)

    def __call__(self, samples: List[str]):
        masked_samples: List[str] = list()
        aug_samples: List[Dict[str, any]] = list()
        for sample in samples:
            if self.keep_base:
                aug_samples.append(sample.copy())

            masked_samples.append(self.masker(sample['text']))

        for idx, pairs in enumerate(zip(masked_samples, samples)):
            masked_sample, sample =  pairs
            sample_copy = sample.copy()
            sample_copy.update({
                'subsetId': int(1e7)+idx,
                'text': masked_sample
            })
            aug_samples.append(sample_copy)

        return aug_samples

class MdaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        tasks: List[str],
        batcher: MdaBatcher,
        batch_size: int = 128,
        augmenter: TextModifier = None,
    ):
        """
        Args:
            data_path (str): Path to the pickled annotations file.
            tasks (List[str]): Tasks under consideration.
            batcher (TripletBatcher): The `Dataloader` to be used.
        """
        super().__init__()
        self.tasks = tasks
        self.data_path = data_path

        self.batcher = batcher
        self.batch_size = batch_size
        self.augmenter = augmenter

    def setup(self, stage: Optional[str] = None, train_fraction: float = 0.88):
        """Read in the data pickle file and perform splitting here.
        Args:
            train_fraction (float): Fraction to use as training data.
        """
        # Read-in the data
        with open(self.data_path, 'rb') as file:
            raw_dataset: Dict[str, Dict[str, any]] = json.load(file)

        # Reformat the json
        # Note: All the og training will be positive indexed
        # New Training data will have negative index
        dataset_base: List[Dict[str, any]] = list()
        dataset_addn: List[Dict[str, any]] = list()
        for key, value in raw_dataset.items():
            if not value['annotation task'] in self.tasks: continue

            if int(key) > 0:
                dataset_base.append({'subsetId': key, **value})
            else:
                dataset_addn.append({'subsetId': key, **value})

        if stage == "fit" or stage is None:
            # Assign train/val dataset indices for use in dataloaders
            # Only selecting indices to avoid data duplication
            train_indices_old, val_indices_old = train_test_split(
                range(len(dataset_base)),
                stratify=[i['hard_label'] for i in dataset_base],
                train_size=train_fraction,
                random_state=44
            )
            if len(dataset_addn) > 0:
                train_indices_new, val_indices_new = train_test_split(
                    range(len(dataset_addn)),
                    stratify=[i['hard_label'] for i in dataset_addn],
                    train_size=train_fraction,
                    random_state=44
                )
            else:
                train_indices_new, val_indices_new = [], []

            # The train dataset
            tr_dataset = [dataset_base[idx] for idx in train_indices_old]
            tr_dataset.extend([dataset_addn[idx] for idx in train_indices_new])

            if self.augmenter:
                tr_dataset = self.augmenter(tr_dataset)

                # Free the graph
                if hasattr(self.augmenter, 'mask_filler'):
                    self.augmenter.mask_filler.model = \
                        self.augmenter.mask_filler.model.cpu()
                    del self.augmenter.mask_filler.model
                    torch.cuda.empty_cache()

            self.train_dataset = MdaDataset(tr_dataset)

            # The val dataset
            val_dataset = [dataset_base[idx] for idx in val_indices_old]
            val_dataset.extend([dataset_addn[idx] for idx in val_indices_new])

            self.val_dataset = MdaDataset(val_dataset)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = MdaDataset(dataset_base)

        if stage == "predict" or stage is None:
            self.pred_dataset = MdaDataset(
                dataset_base,
                inference_mode=True
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.batcher,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.batcher
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.batcher
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_dataset,
            batch_size=self.batch_size,
            collate_fn=self.batcher
        )


class AnnotatorTokenizer:
    def __init__(
        self,
        num_annotators: int,
        pad_token_id: int = 0,
        padding: str = 'ordered',
        infer_token_ids: bool = True
    ):
        """
        Args:
            num_annotators (int): _description_
            pad_token_id (int, optional): _description_. Defaults to 0.
            infer_token_ids (bool, optional): _description_. Defaults to True.
            padding (str, optional): _description_. Defaults to 'ordered'.
        """
        # TODO
        if not (padding == 'ordered' and infer_token_ids):
            raise NotImplementedError("Coming soon")

        self.num_annotators = num_annotators
        self.pad_token_id = pad_token_id

        self._build_ann_to_ids_map()

    def _build_ann_to_ids_map(self):
        self.ann_to_ids = {
            str(i): i for i in range(1, self.num_annotators+1)
        }

        self.ids_to_ann = {
            v: k for k, v in self.ann_to_ids.items()
        }

    def __call__(
        self,
        annotators_batch: List[str],
        # Similar to HF tokenizers `__call__`
        return_tensors: str = 'pt'
    ):
        if not return_tensors == 'pt':
            raise NotImplementedError('Coming soon')

        ann_tokens = []
        for annotators in annotators_batch:
            annotators = annotators.split(',')
            annotators = [self.ann_to_ids[i[-1]] for i in annotators]

            ann_tokens.append(
                [i if i in annotators else 0
                 for i in range(1, self.num_annotators+1)]
            )

        return torch.tensor(ann_tokens, dtype=torch.long)


class CnvAbDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, any]],
        inference_mode: bool = False
    ):
        self.samples: List[Tuple[str, str, str, int,\
            Tuple[float, float]]] = list()

        for sample in samples:
            if inference_mode:
                self.samples.append((
                    sample['subsetId'],
                    sample['text'],
                    sample['annotators'],
                ))
                continue

            self.samples.append((
                sample['subsetId'],
                sample['text'],
                sample['annotators'],
                sample['annotations'],
                (sample['soft_label']['0'], sample['soft_label']['1'])
            ))      

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class CnvAbBatcher:
    def __init__(
        self,
        tnkzr_path: str,
        ann_tknzr: AnnotatorTokenizer,
        has_targets: bool = True,
        use_raw_text: bool = False
    ) -> None:
        """
        Args:
            tnkzr_path (str): Path to load the `Transformers` tokenizer
            to be used.
            ann_tknzr (AnnotatorTokenizer): The `AnnotatorTokenizer` to
            be used.
            has_targets (bool): Does the dataset have target information.
            Defaults to True.
            use_raw_text (bool): Pass text to the tokenizer as-is.
            Defaults to False.
        """
        self.has_targets = has_targets
        self.use_raw_text = use_raw_text
        self.tokenizer = AutoTokenizer.from_pretrained(tnkzr_path)
        self.ann_tknzr = ann_tknzr

    @classmethod
    def _preprocessing(cls, text: str):
        """Split the text into history and `current exchange`.
        Context is the conversation before the `current exchange`.
        """
        # The text is a json string
        conversation: Dict[str, str] = json.loads(text)

        history: List[str] = list()
        curr_exchg: List[str] = list()
        for key, val in conversation.items():
            if key.startswith('prev'): history.extend([key, val])
            else: curr_exchg.extend([key, val])

        delimiter: str = f" "

        return delimiter.join(history), delimiter.join(curr_exchg)

    def __call__(self, batch: Sequence):
        """Use this function as the `collate_fn` mentioned earlier.
        """
        ids = torch.tensor([int(sample[0]) for sample in batch], dtype=torch.int32)

        if self.use_raw_text:
            text_tokens = self.tokenizer(
                [sample[1] for sample in batch], # The text
                max_length=128, # As per https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#fine-tuning
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            )

        # Split up the conversation into context and current exchange
        else:
            conversation: List[Tuple[str, str]] = list(map(
                self._preprocessing,
                [sample[1] for sample in batch]
            ))

            text_tokens = self.tokenizer(
                [item[0] for item in conversation], # The context
                [item[1] for item in conversation], # The Conversation
                max_length=128, # As per https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2#fine-tuning
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            )

        ann_tokens = self.ann_tknzr([sample[2] for sample in batch])

        if not self.has_targets:
            return ids, (text_tokens, ann_tokens)

        # TODO
        targets = "TODO"

        soft_targets = torch.tensor([sample[4] for sample in batch], dtype=torch.float32)

        return ids, (text_tokens, ann_tokens), (targets, soft_targets)


class CnvAbDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        tasks: List[str],
        batcher: CnvAbBatcher,
        batch_size: int = 128
    ):
        """
        Args:
            data_path (str): Path to the pickled annotations file.
            tasks (List[str]): Tasks under consideration.
            batcher (TripletBatcher): The `Dataloader` to be used.
        """
        super().__init__()
        self.tasks = tasks
        self.data_path = data_path

        self.batcher = batcher
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None, train_fraction: float = 0.88):
        """Read in the data pickle file and perform splitting here.
        Args:
            train_fraction (float): Fraction to use as training data.
        """
        # Read-in the data
        with open(self.data_path, 'rb') as file:
            raw_dataset: Dict[str, Dict[str, any]] = json.load(file)

        # Reformat the json
        # Note: All the og training will be positive indexed
        # New Training data will have negative index
        dataset_base: List[Dict[str, any]] = list()
        dataset_addn: List[Dict[str, any]] = list()
        for key, value in raw_dataset.items():
            if not value['annotation task'] in self.tasks: continue

            if int(key) > 0:
                dataset_base.append({'subsetId': key, **value})
            else:
                dataset_addn.append({'subsetId': key, **value})

        if stage == "fit" or stage is None:
            # Assign train/val dataset indices for use in dataloaders
            # Only selecting indices to avoid data duplication
            train_indices_old, val_indices_old = train_test_split(
                range(len(dataset_base)),
                stratify=[i['hard_label'] for i in dataset_base],
                train_size=train_fraction,
                random_state=44
            )
            if len(dataset_addn) > 0:
                train_indices_new, val_indices_new = train_test_split(
                    range(len(dataset_addn)),
                    stratify=[i['hard_label'] for i in dataset_addn],
                    train_size=train_fraction,
                    random_state=44
                )
            else:
                train_indices_new, val_indices_new = [], []

            # The train dataset
            tr_dataset = [dataset_base[idx] for idx in train_indices_old]
            tr_dataset.extend([dataset_addn[idx] for idx in train_indices_new])

            self.train_dataset = CnvAbDataset(tr_dataset)

            # The val dataset
            val_dataset = [dataset_base[idx] for idx in val_indices_old]
            val_dataset.extend([dataset_addn[idx] for idx in val_indices_new])

            self.val_dataset = CnvAbDataset(val_dataset)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = CnvAbDataset(dataset_base)

        if stage == "predict" or stage is None:
            self.pred_dataset = CnvAbDataset(
                dataset_base,
                inference_mode=True
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.batcher,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.batcher
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.batcher
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_dataset,
            batch_size=self.batch_size,
            collate_fn=self.batcher
        )
