def tokenize(llm, prompt, add_special_tokens=True):
    tokens = llm.tokenizer(prompt, add_special_tokens=add_special_tokens)["input_ids"]
    token_str = [s.replace("Ġ","_") for s in llm.tokenizer.convert_ids_to_tokens(tokens)]
    return tokens, token_str