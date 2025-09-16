from lightning import LightningModule
from model.vf import VFEstimator
from model.textencoder import TextEncoder
from tokenizerown import LibriTTSTokenizer
from tokenizer import EmiliaTokenizer
import torch
from einops import rearrange
from utils.mask import min_span_mask, prob_mask_like
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from utils.loss import masked_loss
from utils.utils import write_html
from typing import Any, Dict, Tuple, Union
import math
from einops import repeat
from torchdiffeq import odeint
from model.vocoder import load_vocoder
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only
import random
import numpy as np

class TTSModule(LightningModule):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        attn_dropout: float,
        ff_dropout: float,
        min_span: int,
        voco_type: str,
        sample_rate: int,
        max_audio_len: int,
        optimizer: str = "AdamW",
        lr: float = 1e-4,
        scheduler: str = "linear_warmup_decay",
        use_torchode: bool = True,
        torchdiffeq_ode_method: str = "midpoint",
        torchode_method_klass: str = "tsit5",
        max_steps: int = 200_000,
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
        # self.tokenizer = LibriTTSTokenizer(
        #     special_tokens=["<filler>"],
        #     token_file="./vocab_small.txt",
        #     lowercase=True,
        #     oov_policy="skip",        # 또는 "use_unk", "error"
        #     unk_token="[UNK]",
        # )
        self.tokenizer = EmiliaTokenizer(
            token_file="./vocab.txt",
        )

        # --- Mask/Noise 설정 ---
        self.mask_probs = [0.7, 1.0]   # (fmin, fmax) 비율로 사용
        self.sigma = 1e-6
        self.drop_prob = 0.2
        self.text_drop_prob = 0.1
        self.steps = 32

        # span mask에서 사용하는 파라미터 이름 통일
        self.mask_fracs = tuple(self.mask_probs)
        self.min_span = min_span

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
        self.vocoder = load_vocoder(is_local=False, local_path="", device='cuda')
        # (1) 절대 학습되지 않도록 동결
        if hasattr(self.vocoder, "parameters"):
            for p in self.vocoder.parameters():
                p.requires_grad_(False)
        self.vocoder.eval()  # 드롭아웃/BN 등 비활성화
        self.Ke = 2

    def get_timestep(self, batch_size, dtype, device):
        if random.random()<0.25:
            t = torch.rand((batch_size, ), dtype=dtype, device=device)
        else:
            tnorm = np.random.normal(loc=0, scale=1.0, size=batch_size)
            t = 1 / (1 + np.exp(-tnorm))
            t = torch.tensor(t, dtype=dtype, device=device)
        
        return t

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
        )
        # if span_mask.shape[1] < self.max_audio_len:
        #     span_mask = F.pad(span_mask, (0, self.max_audio_len - span_mask.shape[1]))
        if span_mask.shape[1] > self.max_audio_len:
            span_mask = span_mask[:, :self.max_audio_len]
        return span_mask.bool()

    def forward(self, text, audio, text_mask, audio_mask):
        """
        text: (B, T_text)  - 토큰 ID
        audio: (B, T_audio, n_mels)
        text_mask: (B, T_text)  - 1 유효, 0 패딩
        audio_mask: (B, T_audio) - 1 유효, 0 패딩
        """
        text_emb = self.text_encoder(text)  # (B, T_text, text_emb_dim)

        B = text.shape[0]
        K = int(self.Ke)
        if K > 1:
            text_emb = text_emb.repeat_interleave(K, dim=0)
            audio_mask = audio_mask.repeat_interleave(K, dim=0)
            text_mask = text_mask.repeat_interleave(K, dim=0)
            audio = audio.repeat_interleave(K, dim=0)

        with torch.no_grad():
            span_mask = self.get_span_mask(audio_mask)  # (B, T_audio)

        noise = torch.randn_like(audio)
        # times = torch.rand((B,), dtype=audio.dtype, device=self.device)
        times = self.get_timestep(B*K, audio.dtype, self.device)
        t = rearrange(times, "b -> b () ()")

        # linear blend between x0 and gt (sigma는 최소 섞임 보장)
        x_t = (1 - (1 - self.sigma) * t) * noise + t * audio

        # dropout-style condition mask
        cond_drop_mask = prob_mask_like((B*K, 1), self.drop_prob, self.device)  # (B,1)
        audio_cond_mask = span_mask | cond_drop_mask  # (B, T_audio)

        audio_context = torch.where(
            rearrange(audio_cond_mask, "b l -> b l ()"),
            torch.zeros_like(audio), # True면 여기
            audio, # False면 여기
        )

        if random.random() < self.text_drop_prob:
            text_emb = torch.zeros_like(text_emb, device=text_emb.device)

        # === 예: VFEstimator의 시그니처가 아래와 같다고 가정 ===
        # pred_audio_flow: (B, T_audio, n_mels)
        pred_audio_flow = self.vf_estimator(
            x_t=x_t,
            times=times,
            audio_mask=audio_mask,
            context=audio_context,
            text_embed=text_emb,
            text_mask=text_mask,
        )

        target_audio_flow = audio - (1 - self.sigma) * noise
        loss = masked_loss(pred_audio_flow, target_audio_flow, audio_cond_mask, "mse")
        return loss
    
    @torch.no_grad()
    def solve(self, text, audio, text_mask, audio_mask):
        B = text.shape[0]
        text_emb = self.text_encoder(text)
        
        def fn(t, y):
            times = t.expand(B)

            out = self.vf_estimator.forward_cfg(
                x_t=y,
                context=torch.zeros_like(y),
                times=times,
                text_embed=text_emb,
                audio_mask=audio_mask,
                text_mask=text_mask,
                guidance_scale=3.0,
                concat=False
            )
            return out

        x0 = torch.randn_like(audio)
        t = torch.linspace(0, 1, self.steps, device=self.device)
        # t = repeat(t, "n -> b n", b=B)
        
        trajectory = odeint(
            fn, 
            # torch.compile(fn), 
            x0, 
            t,
            method="rk4",
        )
        sampled = trajectory[-1]

        return sampled

    def training_step(self, batch: Union[Dict[str, torch.Tensor], Tuple], batch_idx: int):
        text, audio, text_mask, audio_mask, original_text = self._parse_batch(batch)
        loss = self.forward(text, audio, text_mask, audio_mask)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.train_step_count += 1

        # 선택: grad clip (Lightning Trainer에서 설정해도 됨)
        if self.grad_clip_val is not None and self.grad_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_val)

        return loss

    @torch.no_grad()
    def validation_step(self, batch: Union[Dict[str, torch.Tensor], Tuple], batch_idx: int):
        text, audio, text_mask, audio_mask, original_text = self._parse_batch(batch)
        loss = self.forward(text, audio, text_mask, audio_mask)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        if batch_idx == 0 and self.current_epoch%10 == 9:
            self.on_sample(batch)
        return {"val_loss": loss}
    
    # @rank_zero_only
    # @torch.no_grad()
    # def on_metric(self, batch):
    #     text, audio, text_mask, audio_mask, original_text = self._parse_batch(batch)   # audio: (B, T, n_mels)

    @rank_zero_only
    @torch.no_grad()
    def on_sample(self, batch):
        import torchaudio as ta
        import numpy as np
        from pathlib import Path
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch

        # 1) 데이터 파싱 & 샘플 생성
        text, audio, text_mask, audio_mask, original_text = self._parse_batch(batch)   # audio: (B, T, n_mels)
        with torch.no_grad():
            sampled = self.solve(text, audio, text_mask, audio_mask)    # (B, T, n_mels)

        # 2) 멜 → 오디오 복원 (Vocos 등: (B, C, T) = (B, n_mels, T) 형태 기대)
        sampled_wavs = self.vocoder.decode(sampled.transpose(1, 2)).detach().cpu()   # (B, T) or (B, 1, T)
        original_wavs = self.vocoder.decode(audio.transpose(1, 2)).detach().cpu()

        step = int(self.global_step)
        B = text.size(0)
        N = min(4, B)

        for i in range(N):
            # 디렉터리 준비: lightning_logs/valid_data/0000010_000/
            base_root = Path(self.trainer.default_root_dir or ".")
            rel_root = Path("valid_data") / f"{step:07d}_{i:03d}"
            base_dir = base_root / rel_root
            (base_dir / "audio").mkdir(parents=True, exist_ok=True)
            (base_dir / "mel").mkdir(parents=True, exist_ok=True)

            # ---- (a) 오디오 저장 ----
            def _to_ch1(w):
                # torchaudio.save는 (C, T) 필요
                if w.ndim == 1:
                    return w.unsqueeze(0)  # (1, T)
                if w.ndim == 2 and w.size(0) > 1:
                    # stereo면 mono로
                    return w.mean(dim=0, keepdim=True)
                return w  # (1, T)
            wav_o = _to_ch1(original_wavs[i]).clamp(-1, 1)
            wav_s = _to_ch1(sampled_wavs[i]).clamp(-1, 1)

            p_wav_o = base_dir / "audio" / "original.wav"
            p_wav_s = base_dir / "audio" / "sampled.wav"
            ta.save(str(p_wav_o), wav_o, sample_rate=self.sample_rate)
            ta.save(str(p_wav_s), wav_s, sample_rate=self.sample_rate)

            # ---- (b) 멜 이미지 저장 ----
            # audio[i], sampled[i] : (T, n_mels) 가정 → 시각화는 (n_mels, T)
            def _save_mel_png(path, mel_tc: torch.Tensor | np.ndarray):
                mel_np = mel_tc.detach().cpu().numpy() if isinstance(mel_tc, torch.Tensor) else mel_tc
                fig = plt.figure(figsize=(8, 3), dpi=200)
                ax = plt.gca()
                ax.imshow(mel_np.T, origin="lower", aspect="auto")
                ax.set_axis_off()
                plt.tight_layout(pad=0)
                fig.savefig(str(path), bbox_inches="tight", pad_inches=0)
                plt.close(fig)

            p_img_o = base_dir / "mel" / "original.png"
            p_img_s = base_dir / "mel" / "sampled.png"
            _save_mel_png(p_img_o, audio[i])
            _save_mel_png(p_img_s, sampled[i])

            # ---- (c) 경로 리스트 구성 (HTML에서 참조할 상대 경로) ----
            audio_paths = [
                str(rel_root / "audio" / "original.wav"),
                str(rel_root / "audio" / "sampled.wav"),
            ]
            image_paths = [
                str(rel_root / "mel" / "original.png"),
                str(rel_root / "mel" / "sampled.png"),
            ]

            # ---- (d) 스크립트 복원 (마스크가 있으면 패딩 제거) ----
            ids = text[i]
            if text_mask is not None:
                ids = ids[text_mask[i]]
            script = original_text[i]
            
            # ---- (e) HTML 생성 & 로깅 ----
            html = write_html(audio_paths, image_paths, script)

            # HTML 파일 저장
            p_html = base_dir / f"{i}th.html"
            with open(p_html, "w", encoding="utf-8") as f:
                f.write(html)

            # 기존 로깅 로직은 그대로 두고,
            try:
                exp = getattr(self.logger, "experiment", None)
                run_id = getattr(self.logger, "run_id", None)

                # MLflow
                if exp is not None and hasattr(exp, "log_text") and run_id is not None:
                    exp.log_text(run_id, html, f"{rel_root}.html")
                    if hasattr(exp, "log_artifacts"):
                        exp.log_artifacts(run_id, str(base_dir), artifact_path=str(rel_root))

                # TensorBoard
                elif exp is not None and hasattr(exp, "add_text"):
                    exp.add_text(f"{rel_root}.html", html, global_step=step)

            except Exception as e:
                print(f"[val log warning] {e}")

    
    def sample(
        self,
        audio_enc,
        audio_mask,
        phoneme,
        phoneme_mask,
    ):
        span_mask = self.get_span_mask(audio_mask)
        span_mask = torch.ones_like(span_mask)

        audio_context = torch.where(rearrange(span_mask, "b l -> b l ()"), 0, audio_enc)

        sampled = self.solve(
            audio_context, audio_mask, phoneme, phoneme_mask
        )
        return sampled
    
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
            original_text = batch["original_text"]
        else:
            text, audio, text_mask, audio_mask = batch

        # 길이 클램프/패딩 정합 책임은 DataModule 쪽이라고 가정
        # 모델 쪽에서는 max_audio_len을 초과하면 잘라줌
        if audio.shape[1] > self.max_audio_len:
            audio = audio[:, : self.max_audio_len]
            audio_mask = audio_mask[:, : self.max_audio_len]

        return text, audio, text_mask, audio_mask, original_text
