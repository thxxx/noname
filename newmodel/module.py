from lightning import LightningModule
from newmodel.vf import VFEstimator
from newmodel.textencoder import TextEncoder
from tokenizerown import LibriTTSTokenizer
import torch
from einops import rearrange
from utils.mask import min_span_mask, prob_mask_like
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from utils.loss import masked_loss
from typing import Any, Dict, Tuple, Union
import math


class TTSModule(LightningModule):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        attn_dropout: float,
        ff_dropout: float,
        kernel_size: int,
        voco_type: str,
        sample_rate: int,
        max_audio_len: int,
        optimizer: str = "AdamW",
        lr: float = 1e-4,
        scheduler: str = "linear_warmup_decay",
        use_torchode: bool = True,
        torchdiffeq_ode_method: str = "midpoint",
        torchode_method_klass: str = "tsit5",
        max_steps: int = 1_000_000,
        n_mels: int = 100,
        text_emb_dim: int = 128,
        downsample_factors: list = (1, 2, 4, 2, 1),
        # 추가 하이퍼파라미터(옵션)
        warmup_ratio: float = 0.05,
        min_lr_ratio: float = 0.1,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.95),
        grad_clip_val: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # --- Models ---
        self.text_encoder = TextEncoder(vocab_size=160, emb_dim=text_emb_dim)
        self.vf_estimator = VFEstimator(
            dim_in=n_mels,
            dim_model=dim,
            conv_hidden=1024,
            num_heads=num_heads,
            Nm=depth,
            downsample_factors=list(downsample_factors),
        )

        # --- Tokenizer (학습 파이프라인에서 텍스트 전처리용) ---
        self.tokenizer = LibriTTSTokenizer(
            special_tokens=["<filler>"],
            token_file="./vocab_small.txt",
            lowercase=True,
            oov_policy="skip",        # 또는 "use_unk", "error"
            unk_token="[UNK]",
        )

        # --- Mask/Noise 설정 ---
        self.mask_probs = [0.7, 1.0]   # (fmin, fmax) 비율로 사용
        self.kernel_size = kernel_size
        self.sigma = 1e-5
        self.drop_prob = 0.1

        # span mask에서 사용하는 파라미터 이름 통일
        self.mask_fracs = tuple(self.mask_probs)
        self.min_span = self.kernel_size

        # --- Misc ---
        self.sample_rate = sample_rate
        self.max_audio_len = max_audio_len
        self.optimizer_name = optimizer
        self.base_lr = lr
        self.max_steps = max_steps
        self.scheduler_name = scheduler
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        self.weight_decay = weight_decay
        self.betas = betas
        self.grad_clip_val = grad_clip_val

        # step 카운터(스케줄러 Lambda 사용 시 필요할 수 있음)
        self.train_step_count = 0

    @torch.compiler.disable
    @torch.no_grad
    def get_span_mask(self, audio_mask: torch.Tensor) -> torch.Tensor:
        """
        audio_mask: (B, L) - 1은 유효, 0은 패딩
        반환: (B, max_audio_len) - True/1인 위치가 mask(가림)
        """
        audio_lens = audio_mask.sum(dim=1).detach().cpu().numpy()
        span_mask = pad_sequence(
            [
                torch.from_numpy(
                    min_span_mask(
                        int(audio_len),
                        fmin=self.mask_fracs[0],
                        fmax=self.mask_fracs[1],
                        min_span=self.min_span,
                    )
                ).to(self.device)
                for audio_len in audio_lens
            ],
            batch_first=True,
        )  # (B, L_real)
        if span_mask.shape[1] < self.max_audio_len:
            span_mask = F.pad(span_mask, (0, self.max_audio_len - span_mask.shape[1]))
        else:
            span_mask = span_mask[:, : self.max_audio_len]
        return span_mask.bool()

    def forward(self, text, audio, text_mask, audio_mask):
        """
        text: (B, T_text)  - 토큰 ID
        audio: (B, T_audio, n_mels)
        text_mask: (B, T_text)  - 1 유효, 0 패딩
        audio_mask: (B, T_audio) - 1 유효, 0 패딩
        """
        B = text.shape[0]
        text_emb = self.text_encoder(text)  # (B, T_text, text_emb_dim)

        with torch.no_grad():
            span_mask = self.get_span_mask(audio_mask)  # (B, T_audio)

        audio_x0 = torch.randn_like(audio)
        times = torch.rand((B,), dtype=audio.dtype, device=self.device)
        t = rearrange(times, "b -> b () ()")

        # linear blend between x0 and gt (sigma는 최소 섞임 보장)
        x_t = (1 - (1 - self.sigma) * t) * audio_x0 + t * audio

        # dropout-style condition mask
        cond_drop_mask = prob_mask_like((B, 1), self.drop_prob, self.device)  # (B,1)
        audio_cond_mask = span_mask | cond_drop_mask  # (B, T_audio)

        audio_context = torch.where(
            rearrange(audio_cond_mask, "b l -> b l ()"),
            torch.zeros_like(audio),
            audio,
        )

        phon_drop_mask = prob_mask_like((B,), self.drop_prob, self.device)  # (B,)
        text_embed = torch.where(
            rearrange(phon_drop_mask, "b -> b () ()"),
            torch.zeros_like(text_emb),
            text_emb,
        )

        # === 예: VFEstimator의 시그니처가 아래와 같다고 가정 ===
        # pred_audio_flow: (B, T_audio, n_mels)
        pred_audio_flow = self.vf_estimator(
            x_t=x_t,
            times=times,
            audio_mask=audio_mask,
            context=audio_context,
            text_embed=text_embed,
            text_mask=text_mask,
        )

        target_audio_flow = audio - (1 - self.sigma) * audio_x0
        loss = masked_loss(pred_audio_flow, target_audio_flow, audio_cond_mask, "mse")
        return loss

    def training_step(self, batch: Union[Dict[str, torch.Tensor], Tuple], batch_idx: int):
        text, audio, text_mask, audio_mask = self._parse_batch(batch)
        loss = self.forward(text, audio, text_mask, audio_mask)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.train_step_count += 1

        # 선택: grad clip (Lightning Trainer에서 설정해도 됨)
        if self.grad_clip_val is not None and self.grad_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_val)

        return loss

    @torch.no_grad()
    def validation_step(self, batch: Union[Dict[str, torch.Tensor], Tuple], batch_idx: int):
        text, audio, text_mask, audio_mask = self._parse_batch(batch)
        loss = self.forward(text, audio, text_mask, audio_mask)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"val_loss": loss}

    def on_train_start(self):
        # 학습 시작 시 lr을 로그에 기록
        for i, pg in enumerate(self.trainer.optimizers[0].param_groups):
            self.log(f"lr/group{i}", pg["lr"], prog_bar=False)

    def on_before_optimizer_step(self, optimizer):
        # step 직전 lr 추적(옵션)
        for i, pg in enumerate(optimizer.param_groups):
            self.log(f"lr/group{i}", pg["lr"], prog_bar=False, on_step=True, on_epoch=False)

    def configure_optimizers(self):
        # Optimizer
        opt_name = self.optimizer_name.lower()
        params = self.parameters()

        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                params, lr=self.base_lr, betas=self.betas, weight_decay=self.weight_decay
            )
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(
                params, lr=self.base_lr, betas=self.betas, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        # Scheduler
        sch_name = self.scheduler_name.lower()
        if sch_name in ["linear_warmup_decay", "cosine_warmup"]:
            total_steps = self.max_steps
            warmup_steps = max(1, int(total_steps * self.warmup_ratio))
            min_lr = self.base_lr * self.min_lr_ratio

            if sch_name == "linear_warmup_decay":
                # warmup → linear decay to min_lr
                def lr_lambda(step):
                    if step < warmup_steps:
                        return float(step + 1) / float(warmup_steps)
                    # decay to min_lr linearly
                    progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
                    scale = 1.0 - progress  # 1 → 0
                    return max(min_lr / self.base_lr, scale)

            else:  # cosine_warmup
                def lr_lambda(step):
                    if step < warmup_steps:
                        return float(step + 1) / float(warmup_steps)
                    progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
                    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return max(min_lr / self.base_lr, cosine)

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": sch_name,
                },
            }

        elif sch_name in ["none", "constant"]:
            return optimizer

        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def _parse_batch(self, batch: Union[Dict[str, torch.Tensor], Tuple]) -> Tuple[torch.Tensor, ...]:
        """
        예상 형태:
          dict 배치: {"text": Long(B,Tt), "audio": Float(B,Ta,D), "text_mask": (B,Tt), "audio_mask": (B,Ta)}
          tuple 배치: (text, audio, text_mask, audio_mask)
        """
        if isinstance(batch, dict):
            text = batch["text"]
            audio = batch["audio"]
            text_mask = batch["text_mask"]
            audio_mask = batch["audio_mask"]
        else:
            text, audio, text_mask, audio_mask = batch

        # 길이 클램프/패딩 정합 책임은 DataModule 쪽이라고 가정
        # 모델 쪽에서는 max_audio_len을 초과하면 잘라줌
        if audio.shape[1] > self.max_audio_len:
            audio = audio[:, : self.max_audio_len]
            audio_mask = audio_mask[:, : self.max_audio_len]

        return text, audio, text_mask, audio_mask
