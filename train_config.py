from utils.misc import dict_to_object

config = dict_to_object({

    # Shared audio parameters
    "audio": {
        "sample_rate": 16000,
        "n_mels": 80,
        "n_fft": 1024,
        "hop_size": 160,
        "win_size": 640,
        "norm_std": 2.2615,
        "norm_mean": -5.8843
    },

    # Tokenizer parameters
    "tokenizer": {
        "silence_token": "<SIL>",
        "unknown_token": "<UNK>",
        "tokens": ['<UNK>', '<SIL>', 'aj', 'aw', 'aː', 'b', 'bʲ', 'c', 'cʰ', 'cʷ', 'd', 'dʒ', 'dʲ', 'd̪', 'ej', 'f', 'fʲ', 'h', 'i', 'iː', 'j', 'k', 'kʰ', 'kʷ', 'l', 'm', 'mʲ', 'm̩', 'n', 'n̩', 'ow', 'p', 'pʰ', 'pʲ', 's', 't', 'tʃ', 'tʰ', 'tʲ', 'tʷ', 't̪', 'v', 'vʲ', 'w', 'z', 'æ', 'ç', 'ð', 'ŋ', 'ɐ', 'ɑ', 'ɑː', 'ɒ', 'ɒː', 'ɔj', 'ə', 'ɚ', 'ɛ', 'ɝ', 'ɟ', 'ɟʷ', 'ɡ', 'ɡʷ', 'ɪ', 'ɫ', 'ɫ̩', 'ɱ', 'ɲ', 'ɹ', 'ɾ', 'ɾʲ', 'ɾ̃', 'ʃ', 'ʉ', 'ʉː', 'ʊ', 'ʎ', 'ʒ', 'ʔ', 'θ']
    },

    # Vocoder parameters
    "vocoder": {
        "resblock": "1",
        "upsample_rates": [5,4,4,2],
        "upsample_kernel_sizes": [11,8,8,4],
        "upsample_initial_channel": 512,
        "resblock_kernel_sizes": [3,7,11],
        "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
    },

    # Duration predictor
    "duration_predictor": {
        "n_embeddings": 512,
        "n_heads": 8,
        "n_layers": 8,
        "n_dim": 512,
        "n_dim_head": 64,
        "n_dim_ffn": 2048,
    },

    # Audio predictor
    "audio_predictor": {
        "n_embeddings": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "n_dim": 1024,
        "n_dim_head": 64,
        "n_dim_ffn": 4096,
    }
})