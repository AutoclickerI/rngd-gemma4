"""Select WikiText-2 train windows used by the original activation captures."""

from fetch import read_selection, sha256


def text_windows(path, tokenizer_path, skip, count, width):
    import pyarrow.parquet as parquet
    from tokenizers import Tokenizer

    if sha256(path.read_bytes()) != read_selection()["wikitext"]["sha256"]:
        raise ValueError("expected the pinned WikiText-2 train parquet")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    bos = tokenizer.token_to_id("<bos>")
    paragraphs = [s for s in parquet.read_table(path, columns=["text"], use_threads=False)["text"].to_pylist()
                  if s.strip()]
    stream = []
    for paragraph in paragraphs[skip:]:
        stream.extend(tokenizer.encode(paragraph, add_special_tokens=False).ids)
        if len(stream) >= count * width:
            break
    if bos is None or len(stream) < count * width:
        raise ValueError("unexpected tokenizer or short WikiText-2 token stream")
    return [[bos] + stream[i * width:(i + 1) * width] for i in range(count)]
