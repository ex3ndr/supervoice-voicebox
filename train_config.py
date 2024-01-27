from utils.misc import dict_to_object

config = dict_to_object({

    # Vocoder parameters
    "vocoder": {

        # Architecture
        "resblock": "1",
        "upsample_rates": [5,4,4,2],
        "upsample_kernel_sizes": [11,8,8,4],
        "upsample_initial_channel": 512,
        "resblock_kernel_sizes": [3,7,11],
        "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],

        # Training
        "training": {
            "segment_size": 8000, # Should be multipliy of sample_rate * 0.01
            "learning_rate": 0.0002,
            "adam_b1": 0.8,
            "adam_b2": 0.99,
            "lr_decay": 0.999,
        }
    },

    # Shared audio parameters
    "audio": {
        "sample_rate": 16000,
        "n_mels": 80,
        "n_fft": 1024,
        "hop_size": 160,
        "win_size": 640
    },

    # dVAE parameters
    "dvae": {
        "tokens": 8192,

        # Audio
        "log_mel_multiplier": -8.0,

        # Architecture
        "codebook_dim": 512,
        "hidden_dim": 512,
        "num_resnet_blocks": 3,
        "kernel_size": 3,
        "num_layers": 2,

        # Training
        "training": {
            "segment_size": 8192,
            "learning_rate": 0.0002,
            "adam_b1": 0.8,
            "adam_b2": 0.99,
            "lr_decay": 0.9999,
        }
    },
    
    # V-Codec parameters
    "vcodec": {

        # Architecture
        "tokens": 8192,
        "groups": 2,
        "quantizers": 8,

        # Training
        "training": {
            "segment_size": 8192,
            "learning_rate": 0.0002,
            "adam_b1": 0.8,
            "adam_b2": 0.99,
            "lr_decay": 0.9999,
        }
    },

    # Voicebox parameters
    "voicebox": {

        # Architecture
        "n_layers": 12,
        "n_embeds": 512,
        "n_heads": 16,

        # Training
        "training": {
            "learning_rate": 0.0002,
            "adam_b1": 0.8,
            "adam_b2": 0.99,
            "lr_decay": 0.9999,
            "segment_size": 8192,
        }
    }
})