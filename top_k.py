import numpy as np

def cos_sim(u, v):
    return np.dot(u, v) / (np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v, v)))


def most_similar(word, k, vocab, word_to_vec):
    distances = []
    for w in vocab:
        if w != word:
            # Calculate cosine similarity between input word vector and current word vector
            similarity = cos_sim(word_to_vec[word], word_to_vec[w])
            distances.append((w, similarity))

    # Sort the cos_sim list by similarity
    distances.sort(key=lambda x: x[1], reverse=True)
    return distances[:k]

def main():
    K = 5
    vecs = np.loadtxt("/Users/eitank/nlp2/ex2/embeddings/wordVectors.txt")

    with open("/Users/eitank/nlp2/ex2/embeddings/vocab.txt", "r", encoding="utf-8") as file:
        vocab = file.readlines()
        vocab = [word.strip() for word in vocab]

    word_to_vec = {word: vec for word, vec in zip(vocab, vecs)}

    for word in ["dog", "england", "john", "explode", "office"]:
        print(f"{word} :", end=" ")
        similar_words = most_similar(word, K, vocab, word_to_vec)
        for i in range(len(similar_words)):
            similar_word = similar_words[i][0]
            similarity = similar_words[i][1]
            print(f"{similar_word} {similarity:.4f}", end="")
            if i == len(similar_words) - 1:
                print()
            else:
                print(", ", end="")


if __name__ == "__main__":
    main()