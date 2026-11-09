"""ImageNet-initialized ResNet-50 with channel/spatial CBAM attention.

This is not WbNet and is not compatible with the 1-D ResNet-9 checkpoints.
CBAM is implemented from its equations, not copied from a research repository.
"""
from __future__ import annotations
from contextlib import nullcontext

import librosa
import torch
from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50

from common.class_names import NUM_CLASSES
from common.config import AudioConfig, ModelConfig


class LogMelFrontend(nn.Module):
    """Batched float32 STFT -> 128 x 163 log-Mel images for default settings.

    The natural time axis is retained, with no image crop, flip, or stretching.
    The three input channels contain the same signal, followed by ImageNet
    channel normalization. Peak-relative dB intentionally discards global gain.
    """

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        mel = librosa.filters.mel(sr=config.sample_rate, n_fft=config.n_fft,
                                  n_mels=config.n_mels, fmin=config.f_min, fmax=config.f_max,
                                  htk=False, norm="slaney")
        if (mel.sum(axis=1) <= 0).any():
            raise ValueError("Empty Mel filters: increase n_fft or reduce n_mels")
        self.register_buffer("mel_filter", torch.tensor(mel, dtype=torch.float32))
        self.register_buffer("window", torch.hann_window(config.n_fft))
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def spectrogram(self, waveform: Tensor) -> Tensor:
        if waveform.ndim != 2 or waveform.shape[1] != self.config.num_samples:
            raise ValueError(f"Expected [batch, {self.config.num_samples}] waveforms; got {tuple(waveform.shape)}")
        context = torch.autocast(waveform.device.type, enabled=False) if waveform.device.type in {"cuda", "cpu"} else nullcontext()
        with context:
            spectrum = torch.stft(waveform.float(), n_fft=self.config.n_fft,
                                  hop_length=self.config.hop_length, window=self.window.float(),
                                  center=True, pad_mode="constant", return_complex=True)
            power = spectrum.real.square() + spectrum.imag.square()
            mel_power = torch.matmul(self.mel_filter.float(), power)
            decibels = 10 * torch.log10(mel_power.clamp_min(1e-10))
            decibels = decibels - decibels.amax(dim=(-2, -1), keepdim=True)
            return decibels.clamp(min=-self.config.top_db, max=0)

    def forward(self, waveform: Tensor, *, augment: bool = False,
                frequency_mask_bins: int = 8, time_mask_frames: int = 8) -> Tensor:
        image = (self.spectrogram(waveform) + self.config.top_db) / self.config.top_db
        if augment:
            image = self._mask(image, frequency_mask_bins, time_mask_frames)
        image = image.unsqueeze(1).expand(-1, 3, -1, -1)
        return (image - self.image_mean) / self.image_std

    @staticmethod
    def _mask(image: Tensor, frequency_width: int, time_width: int) -> Tensor:
        # Vectorized per-example masks; evaluation never calls this path.
        image = image.clone()
        batch, frequencies, frames = image.shape
        for axis, length, maximum in ((1, frequencies, frequency_width), (2, frames, time_width)):
            maximum = min(maximum, length - 1)
            if maximum <= 0:
                continue
            widths = torch.randint(0, maximum + 1, (batch, 1), device=image.device)
            starts = (torch.rand(batch, 1, device=image.device) * (length - widths + 1)).long()
            positions = torch.arange(length, device=image.device).unsqueeze(0)
            mask = (positions >= starts) & (positions < starts + widths)
            image = image.masked_fill(mask.unsqueeze(2) if axis == 1 else mask.unsqueeze(1), 0)
        return image


class CBAM(nn.Module):
    """Sequential channel attention and spatial attention (Woo et al., 2018)."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channel_mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False),
                                         nn.ReLU(inplace=False),
                                         nn.Conv2d(hidden, channels, 1, bias=False))
        self.spatial_conv = nn.Conv2d(2, 1, spatial_kernel, padding=spatial_kernel // 2, bias=False)

    def forward(self, features: Tensor, *, return_attention: bool = False):
        mean = features.mean(dim=(-2, -1), keepdim=True)
        maximum = features.amax(dim=(-2, -1), keepdim=True)
        channel = torch.sigmoid(self.channel_mlp(mean) + self.channel_mlp(maximum))
        attended = features * channel
        spatial_input = torch.cat((attended.mean(dim=1, keepdim=True),
                                   attended.amax(dim=1, keepdim=True)), dim=1)
        spatial = torch.sigmoid(self.spatial_conv(spatial_input))
        result = attended * spatial
        return (result, {"channel": channel, "spatial": spatial}) if return_attention else result


class WingbeatResNet50(nn.Module):
    def __init__(self, audio: AudioConfig, config: ModelConfig, *, load_imagenet: bool | None = None):
        super().__init__()
        self.audio_config, self.model_config = audio, config
        initialize = config.pretrained if load_imagenet is None else load_imagenet
        self.frontend = LogMelFrontend(audio)
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if initialize else None)
        self.backbone.fc = nn.Identity()
        self.attention = nn.ModuleDict({
            str(stage): CBAM({1: 256, 2: 512, 3: 1024, 4: 2048}[stage],
                             config.reduction, config.spatial_kernel)
            for stage in config.attention_stages
        })
        self.classifier = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(2048, NUM_CLASSES))
        self._warmup = False
        self._freeze_bn_stats = False
        self._unfreeze_from_stage = 1

    def set_training_stage(self, *, warmup: bool, unfreeze_from_stage: int = 2,
                           freeze_bn_stats: bool = True) -> None:
        self._warmup = warmup
        self._freeze_bn_stats = freeze_bn_stats
        self._unfreeze_from_stage = unfreeze_from_stage
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if not warmup:
            for stage in range(unfreeze_from_stage, 5):
                for parameter in getattr(self.backbone, f"layer{stage}").parameters():
                    parameter.requires_grad_(True)
        # New CBAM and classifier parameters always learn, including during warmup.
        for parameter in list(self.attention.parameters()) + list(self.classifier.parameters()):
            parameter.requires_grad_(True)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    if self._freeze_bn_stats or not any(p.requires_grad for p in module.parameters()):
                        module.eval()
        return self

    def forward(self, waveform: Tensor, *, return_attention: bool = False,
                augment: bool = False, frequency_mask_bins: int = 8, time_mask_frames: int = 8):
        image = self.frontend(waveform, augment=augment and self.training,
                              frequency_mask_bins=frequency_mask_bins, time_mask_frames=time_mask_frames)
        x = image.contiguous(memory_format=torch.channels_last) if image.is_cuda else image
        x = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
        maps = {}
        for stage in range(1, 5):
            x = getattr(self.backbone, f"layer{stage}")(x)
            if str(stage) in self.attention:
                if return_attention:
                    x, maps[f"stage{stage}"] = self.attention[str(stage)](x, return_attention=True)
                else:
                    x = self.attention[str(stage)](x)
        logits = self.classifier(torch.flatten(self.backbone.avgpool(x), 1))
        return (logits, maps) if return_attention else logits


def build_model(audio: AudioConfig, config: ModelConfig, *, load_imagenet: bool | None = None) -> WingbeatResNet50:
    return WingbeatResNet50(audio, config, load_imagenet=load_imagenet)
