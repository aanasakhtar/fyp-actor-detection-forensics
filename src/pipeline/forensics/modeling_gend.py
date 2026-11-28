"""
Author's native GenD model architecture definition for Hugging Face checkpoints.
Derived from the official WACV 2026 'Deepfake Detection that Generalizes Across Benchmarks' implementation.
"""

from typing import Optional, Union, Any
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = object
    F = None

try:
    from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModel
except ImportError:
    PretrainedConfig = object
    PreTrainedModel = object
    AutoConfig = None
    AutoModel = None


class LinearProbe(nn.Module if torch is not None else object):
    def __init__(self, input_dim: int, num_classes: int = 2, normalize_inputs: bool = False):
        super().__init__()
        if torch is not None:
            self.linear = nn.Linear(input_dim, num_classes)
            self.normalize_inputs = normalize_inputs

    def forward(self, x: Any, **kwargs):
        if self.normalize_inputs and F is not None:
            x = F.normalize(x, p=2, dim=1)
        return self.linear(x)


class CLIPEncoder(nn.Module if torch is not None else object):
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor

        try:
            self._preprocess = CLIPProcessor.from_pretrained(model_name)
        except Exception:
            self._preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        clip: CLIPModel = CLIPModel.from_pretrained(model_name)
        self.vision_model = clip.vision_model
        self.model_name = model_name
        self.features_dim = self.vision_model.config.hidden_size
        self.visual_projection = clip.visual_projection

    def preprocess(self, image: Image.Image) -> Any:
        return self._preprocess(images=image, return_tensors="pt")["pixel_values"][0]

    def forward(self, preprocessed_images: Any) -> Any:
        return self.vision_model(preprocessed_images).pooler_output

    def get_features_dim(self) -> int:
        return self.features_dim


class DINOEncoder(nn.Module if torch is not None else object):
    def __init__(self, model_name: str = "facebook/dinov2-with-registers-base", merge_cls_token_with_patches: Any = None):
        super().__init__()
        import os
        from transformers import AutoImageProcessor, AutoModel
        token = os.environ.get("HF_TOKEN")

        try:
            self._preprocess = AutoImageProcessor.from_pretrained(model_name, token=token)
        except Exception:
            self._preprocess = AutoImageProcessor.from_pretrained("facebook/dinov2-large")

        try:
            self.backbone = AutoModel.from_pretrained(model_name, token=token)
        except Exception:
            # Fallback if gated repo (e.g. facebook/dinov3-vitl16-pretrain-lvd1689m requires Meta auth token)
            self.backbone = AutoModel.from_pretrained("facebook/dinov2-large")

        self.features_dim = self.backbone.config.hidden_size

    def preprocess(self, image: Image.Image) -> Any:
        return self._preprocess(images=image, return_tensors="pt")["pixel_values"][0]

    def forward(self, inputs: Any) -> Any:
        return self.backbone(inputs).last_hidden_state[:, 0]

    def get_features_dim(self) -> int:
        return self.features_dim


class PerceptionEncoder(nn.Module if torch is not None else object):
    def __init__(self, model_name: str = "vit_pe_core_large_patch14_336", img_size: Optional[int] = None):
        super().__init__()
        import timm

        dynamic_img_size = True if img_size is not None else False
        backbone = None
        candidate_names = [
            model_name,
            f"hf-hub:timm/{model_name}.fb",
            f"hf-hub:timm/{model_name}",
            "hf-hub:timm/vit_pe_core_large_patch14_336.fb",
            "vit_large_patch14_336",
        ]
        for name in candidate_names:
            try:
                backbone = timm.create_model(name, pretrained=True, dynamic_img_size=dynamic_img_size)
                break
            except Exception:
                try:
                    backbone = timm.create_model(name, pretrained=False, dynamic_img_size=dynamic_img_size)
                    break
                except Exception:
                    continue

        if backbone is None:
            raise ValueError(f"Could not load any timm architecture for '{model_name}'")

        self.backbone = backbone
        data_config = timm.data.resolve_model_data_config(self.backbone)
        if img_size is not None:
            data_config["input_size"] = (3, img_size, img_size)

        self._preprocess = timm.data.create_transform(**data_config, is_training=False)
        inp = data_config.get("input_size", (3, 224, 224))
        self.input_size = (int(inp[1]), int(inp[2]))
        self.backbone.head = nn.Identity()
        self.features_dim = self.backbone.num_features

    def preprocess(self, image: Image.Image) -> Any:
        return self._preprocess(image)

    def forward(self, inputs: Any) -> Any:
        return self.backbone(inputs)

    def get_features_dim(self) -> int:
        return self.features_dim


class GenDConfig(PretrainedConfig):
    model_type = "GenD"

    def __init__(self, backbone: str = "openai/clip-vit-large-patch14", head: str = "linear", **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.head = head


class GenD(PreTrainedModel):
    config_class = GenDConfig
    base_model_prefix = "gend"
    _tied_weights_keys = []
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    _keys_to_ignore_on_load_unexpected = []

    def __init__(self, config):
        super().__init__(config)
        self._tied_weights_keys = []
        self.head = config.head
        self.backbone = config.backbone
        self.config = config

        self._init_feature_extractor()
        self._init_head()

    @property
    def all_tied_weights_keys(self):
        return []

    def _init_feature_extractor(self):
        backbone = self.backbone
        backbone_lowercase = backbone.lower()

        if "clip" in backbone_lowercase:
            self.feature_extractor = CLIPEncoder(backbone)
        elif "vit_pe" in backbone_lowercase or "pe" in backbone_lowercase:
            img_size = getattr(self.config, "img_size", None)
            if hasattr(self.config, "backbone_args") and self.config.backbone_args:
                img_size = getattr(self.config.backbone_args, "img_size", img_size)
            self.feature_extractor = PerceptionEncoder(backbone, img_size=img_size)
        elif "dino" in backbone_lowercase:
            self.feature_extractor = DINOEncoder(backbone)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def _init_head(self):
        features_dim = self.feature_extractor.get_features_dim()
        if self.head == "linear":
            self.model = LinearProbe(features_dim, 2, normalize_inputs=False)
        elif self.head == "LinearNorm":
            self.model = LinearProbe(features_dim, 2, normalize_inputs=True)
        else:
            self.model = LinearProbe(features_dim, 2, normalize_inputs=False)

    def forward(self, inputs: Any):
        features = self.feature_extractor(inputs)
        outputs = self.model.forward(features)
        return outputs


# Register architecture with Hugging Face AutoConfig and AutoModel
if AutoConfig is not None and AutoModel is not None:
    try:
        AutoConfig.register("GenD", GenDConfig)
        AutoModel.register(GenDConfig, GenD)
    except Exception:
        pass
