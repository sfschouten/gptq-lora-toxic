import transformers


def fix_padding_token(tokenizer: transformers.PreTrainedTokenizer):
    if tokenizer.pad_token is not None:
        # already has pad token
        return tokenizer
    elif tokenizer.eos_token is not None:
        # has <eos> token, re-use as padding token
        tokenizer.pad_token = "[PAD(EOS)]"
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        raise ValueError("No suitable token found to use for padding.")
    return tokenizer
