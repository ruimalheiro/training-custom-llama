DEFAULT_PRETRAINING_MIX = {
    'seed': 42,
    'shard_size': 100_000_000,
    'datasets': {
        'HuggingFaceFW/fineweb-edu': {
            'sample-10BT': {
                'weight': 1.0,
                'transforms': {}
            }
        }
    }
}
