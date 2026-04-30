import random
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd


def tokenize_corpus(corpus: str, n: int = 2) -> list[list[str]]:
    """
    Tokenize a corpus into sentences, each with <s> and </s> boundary tokens.
    Prepends n-1 <s> tokens so that n-gram contexts are well-defined from the
    start of each sentence.

    Args:
        corpus: Raw text that may contain multiple sentences.
        n: The n-gram order, used to determine how many <s> tokens to prepend.

    Returns:
        A list of token lists, each wrapped with (n-1) <s> tokens and one </s>.
    """
    sentences = sent_tokenize(corpus)
    tokenized = []
    for sent in sentences:
        words = word_tokenize(sent.lower())
        # Remove standalone punctuation tokens (e.g. '.', ',') but keep contractions
        words = [w for w in words if any(c.isalpha() for c in w)]
        tokenized.append(["<s>"] * (n - 1) + words + ["</s>"])
    return tokenized


def count_ngrams(tokenized_sentences: list[list[str]], n: int = 2):
    """
    Count n-grams and their (n-1)-gram contexts across all tokenized sentences.
    An n-gram's (n-1)-gram context consists of the immediately preceding n-1 tokens.

    Args:
        tokenized_sentences: List of token lists, each wrapped with <s>/<s> tokens.
        n: The n-gram order (1 = unigram, 2 = bigram, 3 = trigram, …).

    Returns:
        (context_counts, ngram_counts) as defaultdict(int).
        context_counts maps each (n-1)-gram tuple → count.
        ngram_counts maps each n-gram tuple → count.
        For n=1, context_counts maps () → total token count.
    """
    # print(tokenized_sentences)
    ngram_counts = defaultdict(int)
    context_counts = defaultdict(int)

    for tokens in tokenized_sentences:
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngram_counts[ngram] += 1
            prefix = ngram[:-1]  # (n-1)-gram context; () for unigrams
            context_counts[prefix] += 1

    return context_counts, ngram_counts


def ngram_probabilities(corpus: str, n: int = 2, smoothing: str = "none",
                        k: float = 1.0):
    """
    Calculate n-gram probabilities for a corpus of any order n.

    Tokenizes the corpus into sentences with <s>/</s> boundary tokens, then
    counts n-grams and computes conditional probabilities using the chain rule:
        P(w_n | w_1 ... w_{n-1}) = C(w_1..w_n) / C(w_1..w_{n-1})

    Supports:
        - "none"    : MLE      P(w_n | ctx) = C(ngram) / C(ctx)
        - "laplace" : Add-k    P(w_n | ctx) = (C(ngram) + k) / (C(ctx) + k*V)

    Args:
        corpus: Raw text with one or more sentences.
        n: The n-gram order (1 = unigram, 2 = bigram, 3 = trigram, …).
        smoothing: "none" for MLE, "laplace" for add-k smoothing.
        k: Smoothing constant (default 1.0 for standard Laplace).

    Returns:
        (tokenized, ngram_counts, context_counts, probs, vocab_size)
            tokenized     – list of token lists with <s>/</s> boundaries
            ngram_counts  – defaultdict mapping each n-gram tuple → count
            context_counts – defaultdict mapping each (n-1)-gram prefix → count
            probs         – dict mapping each n-gram tuple → probability
            vocab_size    – number of unique tokens (V)
    """
    tokenized = tokenize_corpus(corpus, n=n)

    # Collect all tokens to build vocabulary
    all_tokens = [tok for sent in tokenized for tok in sent]
    vocab = sorted(set(all_tokens))
    V = len(vocab)

    context_counts, ngram_counts = count_ngrams(tokenized, n)

    # Compute probabilities
    probs = {}
    if smoothing == "laplace":
        for ngram, count in ngram_counts.items():
            prefix = ngram[:-1]
            probs[ngram] = (count + k) / (context_counts[prefix] + k * V)
    else:
        for ngram, count in ngram_counts.items():
            prefix = ngram[:-1]
            probs[ngram] = count / context_counts[prefix]

    return tokenized, ngram_counts, context_counts, probs, V


def print_tables(corpus: str, n: int = 2, smoothing: str = "none", k: float = 1.0) -> None:
        
    tokenized, ngram_counts, context_counts, probs, V = ngram_probabilities(
        corpus, n=n, smoothing=smoothing, k=k)

    # Build count_table: count_table[ctx_tuple][last_word] = C(ngram)
    count_table = defaultdict(lambda: defaultdict(int))
    for ngram, count in ngram_counts.items():
        count_table[ngram[:-1]][ngram[-1]] = count

    ngram_name = {1: "UNIGRAM", 2: "BIGRAM", 3: "TRIGRAM"}.get(n, f"{n}-GRAM")

    # --- 2D n-gram count table ---
    all_words = sorted({w for row in count_table.values() for w in row})
    ctx_labels = [" ".join(ctx) if ctx else "(total)" for ctx in sorted(count_table)]
    count_df = pd.DataFrame(
        [{w: count_table[ctx].get(w, 0) for w in all_words} for ctx in sorted(count_table)],
        index=ctx_labels
    )
    count_df.index.name = "context"
    print(f"\n{ngram_name} COUNT TABLE  (rows = context, columns = last word)")
    print(count_df.to_string())
    print()

    # --- N-gram probability table ---
    last = f"w{n}"
    ctx_vars = " ".join(f"w{i}" for i in range(1, n))  # "" for n=1
    ngram_vars = " ".join(f"w{i}" for i in range(1, n + 1))
    ctx_str = f"|{ctx_vars}" if ctx_vars else ""

    if smoothing == "laplace":
        print(f"{ngram_name} PROBABILITIES  —  Laplace (add-{k})")
        print(f"  P({last}{ctx_str}) = (C({ngram_vars}) + {k}) / (C({ctx_vars or 'total'}) + {k}*V)")
    else:
        print(f"{ngram_name} PROBABILITIES  —  MLE (no smoothing)")
        print(f"  P({last}{ctx_str}) = C({ngram_vars}) / C({ctx_vars or 'total'})")

    rows = []
    for ngram in sorted(probs):
        c_ngram = ngram_counts[ngram]
        c_ctx = context_counts[ngram[:-1]]
        row = {"ngram": f"({', '.join(ngram)})", "C(ngram)": c_ngram, "C(ctx)": c_ctx}
        if smoothing == "laplace":
            row["C+k"] = c_ngram + k
            row["C+kV"] = c_ctx + k * V
        row["P"] = probs[ngram]
        rows.append(row)

    prob_df = pd.DataFrame(rows).set_index("ngram")
    print(prob_df.to_string(float_format="{:.4f}".format))
    print()


def generate_sentence(corpus: str, n: int = 2, smoothing: str = "none", k: float = 1.0,
                      max_length: int = 20, seed: int | None = None) -> str:
    """
    Generate a sentence by sampling from n-gram probabilities.

    Maintains a context of the last n-1 tokens, starting from ("<s>",) * (n-1).
    Repeatedly samples the next word until </s> is produced or max_length is reached.

    Args:
        corpus: Raw text used to build the n-gram model.
        n: The n-gram order (1 = unigram, 2 = bigram, 3 = trigram, …).
        smoothing: "none" for MLE, "laplace" for add-k smoothing.
        k: Smoothing constant (used only when smoothing="laplace").
        max_length: Maximum number of words before forcing a stop.
        seed: Optional random seed for reproducibility.

    Returns:
        The generated sentence as a string.
    """
    if seed is not None:
        random.seed(seed)

    tokenized, ngram_counts, context_counts, probs, V = ngram_probabilities(
        corpus, n=n, smoothing=smoothing, k=k)

    vocab = sorted({tok for sent in tokenized for tok in sent})

    # Build lookup: context tuple -> list of (next_word, probability)
    next_word_dist = defaultdict(list)
    for ngram, prob in probs.items():
        next_word_dist[ngram[:-1]].append((ngram[-1], prob))

    # For Laplace, add unseen n-grams so any vocab word can be sampled.
    if smoothing == "laplace":
        for ctx, ctx_count in context_counts.items():
            seen = {w for w, _ in next_word_dist[ctx]}
            for w in vocab:
                if w not in seen:
                    next_word_dist[ctx].append((w, k / (ctx_count + k * V)))

    # --- Sample sentence ---
    words = []
    context = ("<s>",) * (n - 1)  # empty tuple for unigrams

    for _ in range(max_length):
        candidates = next_word_dist[context]
        if not candidates:
            break

        tokens, weights = zip(*candidates)
        chosen = random.choices(tokens, weights=weights, k=1)[0]

        if chosen == "</s>":
            break

        words.append(chosen)
        context = (*context[1:], chosen)  # slide context window forward

    return " ".join(words)


if __name__ == "__main__":
    corpus = "I am Sam. Sam I am. I do not like green eggs and ham."

    # MLE (no smoothing)
    print_tables(corpus, smoothing="none")

    # Laplace smoothing (add-1)
    print_tables(corpus, smoothing="laplace", k=1)

    # --- Generate sentences ---
    print("=" * 70)
    print("GENERATED SENTENCES")
    print("=" * 70)

    print("\n  MLE (no smoothing):")
    for i in range(5):
        sent = generate_sentence(corpus, smoothing="none", seed=i)
        print(f"    {i+1}. {sent}")

    print("\n  Laplace (add-1):")
    for i in range(5):
        sent = generate_sentence(corpus, smoothing="laplace", k=1, seed=i)
        print(f"    {i+1}. {sent}")
    print()

    # --- Demonstrate ngram_probabilities for n = 1, 2, 3 ---
    print("\n" + "#" * 70)
    print("# ngram_probabilities() — with sentence boundaries")
    print("#" * 70 + "\n")

    for n in (1, 2, 3):
        print_ngram_tables(corpus, n, smoothing="none")

    # Bigrams with Laplace
    print_ngram_tables(corpus, n=2, smoothing="laplace", k=1)
