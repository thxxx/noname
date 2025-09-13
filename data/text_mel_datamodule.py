import random
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import torch
import torchaudio as ta
import pandas as pd
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from tokenizerown import LibriTTSTokenizer

from utils.feature import TorchAudioFbank, TorchAudioFbankConfig


class TextMelDataModule(LightningDataModule):
    def __init__(
        self,
        name,
        dataset,
        batch_size,
        num_workers,
        pin_memory,
        n_spks,
        n_fft,
        n_feats,          # == n_mels
        sample_rate,
        hop_length,
        f_min,
        f_max,
        data_statistics,
        seed,
        load_durations,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # alias for clarity
        self.hparams.n_mels = n_feats

    def setup(self):
        hp = self.hparams

        self.trainset = TextMelDataset(
            dataset=hp.dataset['train'],
            n_spks=hp.n_spks,
            n_fft=hp.n_fft,
            n_mels=hp.n_mels,
            sample_rate=hp.sample_rate,
            hop_length=hp.hop_length,
            f_min=hp.f_min,
            f_max=hp.f_max,
            data_parameters=hp.data_statistics,
            seed=hp.seed,
            load_durations=hp.load_durations,
        )

        self.validset = TextMelDataset(
            dataset=hp.dataset['validation'],
            n_spks=hp.n_spks,
            n_fft=hp.n_fft,
            n_mels=hp.n_mels,
            sample_rate=hp.sample_rate,
            hop_length=hp.hop_length,
            f_min=hp.f_min,
            f_max=hp.f_max,
            data_parameters=hp.data_statistics,
            seed=hp.seed,
            load_durations=hp.load_durations,
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
        )

    def val_dataloader(self):
        return DataLoader(
            self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
        )

    def teardown(self, stage: Optional[str] = None):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        n_spks: int,
        n_fft: int = 1024,
        n_mels: int = 100,
        sample_rate: int = 22050,
        hop_length: int = 256,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        data_parameters: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
        load_durations: bool = False,
    ):
        # df = pd.read_csv(filelist_path, sep="|", header=None)
        # self.filepaths_and_text = [[df.iloc[i][0], df.iloc[i][1]] for i in range(len(df))]
        self.dataset = dataset
        self.n_spks = n_spks

        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.load_durations = load_durations
        self.sample_rate = sample_rate
        self.tokenizer = LibriTTSTokenizer(
            special_tokens=["<filler>"],
            token_file="/home/khj6051/star/vocab_small.txt",
            lowercase=True,
            oov_policy="skip",        # OOV은 버림 (또는 "use_unk", "error")
            unk_token="[UNK]",        # oov_policy="use_unk"일 때만 필요
        )

        self.fbank = TorchAudioFbank(
            config=TorchAudioFbankConfig(
                sampling_rate=sample_rate,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
            )
        )

        if data_parameters is None:
            data_parameters = {"mel_mean": 0.0, "mel_std": 1.0}
        self.data_parameters = data_parameters

        if seed is not None:
            random.seed(seed)
        # random.shuffle(self.filepaths_and_text)

    def _load_durations(self, wav_path: Path, text_tensor: torch.Tensor) -> torch.Tensor:
        data_dir = wav_path.parent.parent
        name = wav_path.stem
        dur_loc = data_dir / "durations" / f"{name}.npy"

        try:
            durs = torch.from_numpy(np.load(dur_loc).astype(int))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Durations not found at {dur_loc}. "
                "Generate durations first: python matcha/utils/get_durations_from_trained_model.py"
            ) from e

        assert len(durs) == len(text_tensor), f"Dur len {len(durs)} != text len {len(text_tensor)}"
        return durs

    def _get_mel(self, audio) -> torch.Tensor:
        # audio, sr = ta.load(filepath)  # (C, T)
        # if sr != self.sample_rate:
        #     raise ValueError(f"Sample rate mismatch: got {sr}, expected {self.sample_rate}")
        mel = self.fbank.extract(torch.from_numpy(audio).float(), self.sample_rate).squeeze(0)  # (n_mels, T')
        # normalize
        mel = (mel - float(self.data_parameters["mel_mean"])) / float(self.data_parameters["mel_std"])
        return mel  # (n_mels, T')

    def __getitem__(self, index: int):
        data = self.dataset[index]
        # filepath, spk, text = self._resolve_fields(self.filepaths_and_text[index])
        mel = self._get_mel(data['audio']['array'])
        token_ids = self.tokenizer.texts_to_token_ids([data['text']])[0]

        duration = data['audio']['array'].shape[-1]/self.sample_rate

        return {"x": torch.tensor(token_ids), "y": torch.tensor(mel).transpose(1, 0), "filepath": "filepath", "durations":None}
        # return {"x": torch.tensor(token_ids), "y": torch.tensor(mel).transpose(1, 0), "filepath": "filepath", "durations": torch.tensor(duration).unsqueeze(0)}

    def __len__(self):
        return len(self.dataset)


class TextMelBatchCollate:
    def __init__(self, n_spks: int):
        self.n_spks = n_spks

    def __call__(self, batch):
        B = len(batch)
        y_lengths = [item["y"].shape[-1] for item in batch]
        x_lengths = [item["x"].shape[-1] for item in batch]

        y_max_len = max(y_lengths)
        x_max_len = max(x_lengths)
        n_mels = batch[0]["y"].shape[-2]

        # Padded containers
        y = torch.zeros((B, n_mels, y_max_len), dtype=torch.float32)
        x = torch.zeros((B, x_max_len), dtype=torch.long)

        # Optional durations (token-level)
        has_durations = batch[0]["durations"] is not None
        durations = None
        if has_durations:
            durations = torch.zeros((B, x_max_len), dtype=torch.long)

        filepaths: List[str] = []

        for i, item in enumerate(batch):
            y_i, x_i = item["y"], item["x"]
            y[:, :, : y_i.shape[-1]][i] = y_i
            x[i, : x_i.shape[-1]] = x_i
            filepaths.append(item["filepath"])

            if has_durations and item["durations"] is not None:
                durations[i, : item["durations"].shape[-1]] = item["durations"]

        y_lengths = torch.tensor(y_lengths, dtype=torch.long)
        x_lengths = torch.tensor(x_lengths, dtype=torch.long)

        # Attention masks (True = valid, False = pad)
        x_mask = (torch.arange(x_max_len)[None, :].expand(B, -1) < x_lengths[:, None])
        y_mask = (torch.arange(y_max_len)[None, :].expand(B, -1) < y_lengths[:, None])

        batch_out = {
            "text": x,                              # (B, Lx)
            "text_mask": x_mask.to(torch.bool),     # (B, Lx)
            "audio": y,                              # (B, n_mels, Ly)
            "audio_mask": y_mask.to(torch.bool),     # (B, Ly)
        }

        return batch_out
