# define hyperparameters
model_cfg = {
    'embed_dim': 768,
    'vision_cfg': {
        'image_size': 224,
        'layers': [3, 4, 6, 3],
        'width': 64,
        'patch_size': None,
        'head_width': 64
    },
    'text_cfg': {
        'bert_model_name': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
        'context_length': 77,
        'vocab_size': 30522
    }
}