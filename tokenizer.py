import tiktoken

from pathlib import Path
from collections import defaultdict
from tiktoken.load import load_tiktoken_bpe


class Tokenizer:
    def __init__(self, path):
        self.special_tokens = defaultdict(int)
        self.num_reserved_special_tokens = 256
        self.pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

        self.vocab = load_tiktoken_bpe(path)
        self.num_base_tokens = len(self.vocab)

        special_tokens = [
            '<|begin_of_text|>',
            '<|end_of_text|>',
            '<|reserved_special_token_0|>',
            '<|reserved_special_token_1|>',
            '<|reserved_special_token_2|>',
            '<|reserved_special_token_3|>',
            '<|start_header_id|>',
            '<|end_header_id|>',
            '<|reserved_special_token_4|>',
            '<|eot_id|>',
        ]

        special_tokens += [
            f'<|reserved_special_token_{i}|>'
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]

        self.special_tokens = {
            token: self.num_base_tokens + i for i, token in enumerate(special_tokens)
        }

        self.model = tiktoken.Encoding(
            name=Path(path).name,
            pat_str=self.pat_str,
            mergeable_ranks=self.vocab,
            special_tokens=self.special_tokens
        )

        self.vocab_size = self.model.n_vocab

        self.bos_id = self.special_tokens['<|begin_of_text|>']
        self.eos_id = self.special_tokens['<|end_of_text|>']
        self.pad_id = self.special_tokens['<|reserved_special_token_0|>']
        self.stop_tokens = {
            self.special_tokens['<|end_of_text|>'],
            self.special_tokens['<|eot_id|>'],
        }
        
    def encode(
        self,
        s,
        *,
        bos=False,
        eos=False,
        allowed_special=set(),
        disallowed_special=()
    ):
        tokens = self.model.encode(s, allowed_special=allowed_special, disallowed_special=disallowed_special)
        if bos:
            tokens.insert(0, self.bos_id)
        if eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, t):
        return self.model.decode(t)

    def encode_instruct(self, s, system_msg=True):
        bot = self.special_tokens['<|begin_of_text|>']
        sh = self.special_tokens['<|start_header_id|>']
        eh = self.special_tokens['<|end_header_id|>']
        eot = self.special_tokens['<|eot_id|>']

        tokens = [bot, sh]
        if system_msg:
            tokens.extend(self.encode('system'))
            tokens.extend([eh])
            tokens.extend(self.encode('\n' + 'You are a helpful AI assistant'))
            tokens.extend([eot, sh])
        tokens.extend(self.encode('user'))
        tokens.extend([eh])
        tokens.extend(self.encode('\n'))
        tokens.extend(self.encode(s))
        tokens.extend([eot, sh])
        tokens.extend(self.encode('assistant'))
        tokens.extend([eh])
        tokens.extend(self.encode('\n'))
        return tokens
