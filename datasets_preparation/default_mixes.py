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

DEFAULT_INSTRUCT_MIX = {
    'seed': 42,
    'datasets': {
        'HuggingFaceH4/ultrachat_200k': {
            'default': {
                'weight': 1.0,
                'transforms': {
                    'max_turns': 8
                }
            }
        },
        'lmsys/lmsys-chat-1m': {
            'default': {
                'weight': 0.0,
                'transforms': {
                    'max_turns': 8,
                    'placeholders': {
                        'replace': False,
                        'random_options': [
                            'Alex',
                            'Sam',
                            'Taylor',
                            'Jordan',
                            'Casey',
                            'Riley',
                            'Jamie',
                            'Morgan',
                            'Avery',
                            'Quinn',
                            'Charlie',
                            'Drew',
                            'Elliot',
                            'Rowan',
                            'Skyler',
                            'Devon',
                            'Cameron',
                            'Robin',
                            'Sage',
                            'Reese',
                            'Peyton',
                            'Kai',
                            'Jesse'
                        ]
                    }
                }
            }
        }
    }
}

DEFAULT_DPO_MIX = {
    'seed': 42,
    'datasets': {
        'Anthropic/hh-rlhf': {
            'default': {
                'weight': 1.0,
                'transforms': {}
            }
        }
    }
}
