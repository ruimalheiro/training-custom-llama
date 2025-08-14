import tiktoken
import os

from config import config
os.environ['HF_HOME'] = config.hf_home
os.environ['HF_DATASETS_CACHE'] = f'{config.hf_home}/datasets'
os.environ['HF_HUB_CACHE'] = f'{config.hf_home}/hub'

from abc import ABC, abstractmethod
from typing import Iterable, List
from pathlib import Path
from collections import defaultdict
from tiktoken.load import load_tiktoken_bpe
from transformers import AutoTokenizer


class BaseTokenizer(ABC):

    @abstractmethod
    def encode(self, text, *, bos=False, eos=False, allowed_special=set(), disallowed_special=()):
        ...

    @abstractmethod
    def decode(self, ids):
        ...

    def encode_instruct(self, s, system_msg=True):
        bot = self.bos_id
        sh = self.sh_id
        eh = self.eh_id
        eot = self.eot_id

        tokens = [bot, sh]
        if system_msg:
            tokens.extend(self.encode('system'))
            tokens.extend([eh])
            tokens.extend(self.encode('\n' + config.system_prompt))
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

class TikTokenizer(BaseTokenizer):
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
        self.sh_id = self.special_tokens['<|start_header_id|>']
        self.eh_id = self.special_tokens['<|end_header_id|>']
        self.eot_id = self.special_tokens['<|eot_id|>']
        self.pad_id = self.special_tokens['<|reserved_special_token_0|>']

        self.stop_tokens = {
            self.eos_id,
            self.eot_id,
        }
        
    def encode(
        self,
        text,
        *,
        bos=False,
        eos=False,
        allowed_special=set(),
        disallowed_special=()
    ):
        tokens = self.model.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special)
        if bos: tokens.insert(0, self.bos_id)
        if eos: tokens.append(self.eos_id)
        return tokens

    def decode(self, tokens):
        return self.model.decode(tokens)

class HFTokenizer(BaseTokenizer):
    def __init__(self, path):
        self.num_reserved_special_tokens = 256
        self.model = AutoTokenizer.from_pretrained(path, token=config.hf_token)
        self.model.model_max_length = int(1e30)

        update_tokens = []
        for token in [
            '<|begin_of_text|>',
            '<|end_of_text|>',
            '<|start_header_id|>',
            '<|end_header_id|>',
            '<|eot_id|>',
            '<|reserved_special_token_0|>'
        ]:
            if token not in self.model.get_vocab():
                update_tokens.append(token)

        if update_tokens:
            self.model.add_special_tokens({'additional_special_tokens': update_tokens})

        self.model.bos_token = '<|begin_of_text|>'
        self.model.bos_token_id = self.model.convert_tokens_to_ids(self.model.bos_token)
        self.model.eos_token = '<|end_of_text|>'
        self.model.eos_token_id = self.model.convert_tokens_to_ids(self.model.eos_token)
        self.model.pad_token = '<|reserved_special_token_0|>'
        self.model.pad_token_id = self.model.convert_tokens_to_ids(self.model.pad_token)

        self.vocab_size = len(self.model)

        self.bos_id = self.model.bos_token_id
        self.eos_id = self.model.eos_token_id
        self.sh_id = self.model.convert_tokens_to_ids('<|start_header_id|>')
        self.eh_id = self.model.convert_tokens_to_ids('<|end_header_id|>')
        self.eot_id = self.model.convert_tokens_to_ids('<|eot_id|>')
        self.pad_id = self.model.pad_token_id

        self.stop_tokens = {
            self.eos_id,
            self.eot_id,
        }

    def encode(
        self,
        text,
        *,
        bos=False,
        eos=False,
        **kw
    ):
        tokens = self.model.encode(text, add_special_tokens=False)
        if bos: tokens.insert(0, self.bos_id)
        if eos: tokens.append(self.eos_id)
        return tokens

    def decode(self, tokens):
        return self.model.decode(tokens, skip_special_tokens=False)

def init_tokenizer(checkpoint_path, huggingface_tokenizer=False):
    if huggingface_tokenizer:
        return HFTokenizer(checkpoint_path)
    return TikTokenizer(checkpoint_path)
