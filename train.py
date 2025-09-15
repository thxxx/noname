import time
from pathlib import Path
from typing import cast
import torch
from lightning import Callback, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from datasets import load_dataset

from model.module import TTSModule
from data.text_mel_datamodule import TextMelDataModule

def main():
    torch._dynamo.config.optimize_ddp = False
    seed_everything(42, workers=True)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high") # high = 정밀도와 속도의 균형. default 값
    
    ds = load_dataset("atmansingh/ljspeech")

    batch_size = 128
    model_dim = 256
    n_mels = 100
    text_emb_dim = 128
    max_audio_len = 2000
    learning_rate = 1e-4

    datamodule = TextMelDataModule(
        name="ljspeech",
        dataset=ds,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
        n_spks=1,             # LJSpeech = 단일 화자 → 1
        n_fft=1024,
        n_feats=100,           # mel bins
        sample_rate=24000,
        hop_length=256,
        f_min=0,
        f_max=12000,
        data_statistics={"mel_mean": 0.0, "mel_std": 1.0},
        seed=42,
        load_durations=False, # alignment 정보 필요 없으면 False
    )
    
    model = TTSModule(
        dim=model_dim,
        depth=5,
        num_heads=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        min_span=10,
        voco_type='vocos',
        sample_rate=24000,
        max_audio_len=max_audio_len,
        optimizer = "AdamW",
        lr = learning_rate,
        scheduler = "linear_warmup_decay",
        use_torchode = True,
        torchdiffeq_ode_method = "midpoint",
        torchode_method_klass = "tsit5",
        max_steps = 10_000,
        n_mels = n_mels,
        text_emb_dim = text_emb_dim,
        downsample_factors = [1, 2, 4, 2, 1],
        # 추가 하이퍼파라미터(옵션)
        warmup_ratio = 0.005,
        min_lr_ratio = 0.1,
        weight_decay = 0.01,
        betas = (0.9, 0.95),
        grad_clip_val = 1.0,
    )

    # PyTorch Lightning에서 사용하는 example_input_array. model의 forward 입력값을 확실히 알려주는 용도
    model.example_input_array = (
        torch.randint(0, 160, (batch_size, 150), dtype=torch.long), # text token_ids
        torch.randn(batch_size, max_audio_len, n_mels), # audio mel-spectrogram
        torch.ones(batch_size, 150, dtype=torch.bool), # text mask
        torch.ones(batch_size, max_audio_len, dtype=torch.bool), # audio mask
    )

    # 로깅방식
    Path("logs").mkdir(exist_ok=True) # logs 폴더를 만드는데 이미 있어도 에러없이 지나간다.
    logger = TensorBoardLogger(save_dir="logs", name="project1")
    run_name = time.strftime("%Y-%m-%d_%H-%M-%S")

    if run_name is not None:
        checkpoint_dir = Path("checkpoints") / str(run_name) # 나누기가 아니라 dir 추가하는거구나
        checkpoint_dir.mkdir(parents=True, exist_ok=True) # 폴더가 여러개 필요해도 만든다.
    else:
        checkpoint_dir = None

    if checkpoint_dir is not None:
        with open(checkpoint_dir / "model.txt", "w") as f:
            f.write(str(model))
    
    # Pytorch lightning의 callback들 : 자동으로 특정 타이밍에 실행되는 로직.
    model_checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{step:07d}-{val/loss:.4f}",
        monitor="val/loss", # 이 metric을 기준으로
        # every_n_train_steps=5000,
        # save_top_k=5,       # 가장 좋은 모델 3개 저장
        # mode="min",         # 낮을 수록 좋다.
        auto_insert_metric_name=False,
        save_top_k=-1,           # 모든 체크포인트 저장
        every_n_train_steps=500,  # 1000 스텝마다 저장
        save_on_train_epoch_end=False,  # 스텝 단위로 저장할 거면 이걸 False로
    )
    callbacks: list[Callback] = [model_checkpoint]
    callbacks.append(LearningRateMonitor(logging_interval="step")) # lr도 step마다 로깅해라!
    
    precision = "32"
    assert precision == "16-mixed" or precision == "32" or precision == "bf16-mixed"

    trainer = Trainer(
        # strategy="ddp",
        strategy=DDPStrategy(find_unused_parameters=True),
        accelerator="gpu",
        devices=4,
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        detect_anomaly=False,
        fast_dev_run=False,
        logger=logger,
        log_every_n_steps=5, # original : 10
        max_steps=10_000, # default : 1M
        val_check_interval=1.0,
        num_sanity_val_steps=3,
        precision=precision,
    )

    trainer.fit(model=model, datamodule=datamodule, ckpt_path=None)

    if checkpoint_dir is None:
        save_path = "model.ckpt"
    else:
        save_path = checkpoint_dir / "model.ckpt"

    if False:
        trainer.test(model=model, datamodule=datamodule)
    else:
        trainer.test(ckpt_path="best", datamodule=datamodule)
        trainer.save_checkpoint(save_path, weights_only=True)


if __name__ == "__main__":
    main()