from data import CLIP_SECONDS, HOP_LENGTH, N_MELS, SAMPLE_RATE

VIT_CONFIGS = {
    "base": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "intermediate_size": 4 * 768,
        "num_attention_heads": 12,
        "attention_probs_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.1,
        "qkv_bias": True,
        "initializer_range": 0.02,
    },
    "large": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "intermediate_size": 4 * 1024,
        "num_attention_heads": 16,
        "attention_probs_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.1,
        "qkv_bias": True,
        "initializer_range": 0.02,
    },
}


def transformer_config(vit_size):
    return dict(VIT_CONFIGS[vit_size])


def audio_config():
    spec_time = SAMPLE_RATE * CLIP_SECONDS // HOP_LENGTH + 1
    return {
        "spectrogram_size": (N_MELS, spec_time),
        "patch_size": (16, 16),
        "patch_stride": (16, 16),
        "num_channels": 1,
    }


def video_config(num_frames=16, frame_size=224):
    return {
        "num_frames": num_frames,
        "tubelet_size": 2,
        "image_size": frame_size,
        "num_channels": 3,
        "patch_size": 16,
    }
