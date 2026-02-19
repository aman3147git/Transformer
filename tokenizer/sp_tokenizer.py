import sentencepiece as spm

def train_tokenizer():
    spm.SentencePieceTrainer.train(
        input="data/tiny_corpus.txt",
        model_prefix="tokenizer/sp",
        vocab_size=25,
        model_type="unigram",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )

    sp = spm.SentencePieceProcessor()
    sp.load("tokenizer/sp.model")

    text = "i love playing cricket"

    ids = sp.encode(text)
    tokens = sp.encode(text, out_type=str)

    print("Text:", text)
    print("Tokens:", tokens)
    print("IDs:", ids)
    decoded = sp.decode(ids)
    print(decoded)

if __name__ == "__main__":
    train_tokenizer()
