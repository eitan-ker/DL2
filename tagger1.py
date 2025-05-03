import os
import sys
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


class Mlp(nn.Module):
    def __init__(self, vocabulary_size, hidden_size, num_classes, embedding_size=50, context_size=5):
        super(Mlp, self).__init__()

        # word embedding layer
        self.word_embeddings = nn.Embedding(vocabulary_size, embedding_size)

        # neural network layers
        self.hidden_layer = nn.Linear(embedding_size * context_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_classes)

        # activation function
        self.activation = nn.Tanh()

    def forward(self, x):
        # get embeddings and flatten (250 because 5 words of 50 dim each)
        embeddings = self.word_embeddings(x).view(-1, 250)

        # apply first layer with activation
        hidden = self.activation(self.hidden_layer(embeddings))

        # apply dropout for regularization
        hidden = F.dropout(hidden, training=self.training)

        # apply final layer
        logits = self.output_layer(hidden)

        # return softmax prob
        return F.log_softmax(logits, dim=1)


TASK_TYPE = str(sys.argv[1])
BATCH_SIZE = 64
if TASK_TYPE == 'pos':
    FIELD_SEPARATOR = ' '
    VALIDATION_BATCH_SIZE = 64
    TRAINING_EPOCHS = 12
    IS_ENTITY_RECOGNITION = False
elif TASK_TYPE == 'ner':
    FIELD_SEPARATOR = '\t'
    VALIDATION_BATCH_SIZE = 64
    TRAINING_EPOCHS = 12
    IS_ENTITY_RECOGNITION = True
else:
    print("invalid choice should run 'python3 tagger1.py ner/pos'")
    exit(1)


def train_model(model, optimizer, train_data, eval_data, num_epochs, tag_decoder):
    eval_accuracy_history = []
    eval_loss_history = []

    for epoch in range(num_epochs):
        model.train()

        # total loss init
        epoch_loss = 0.0
        for batch_num, (inputs, targets) in enumerate(train_data):
            optimizer.zero_grad()

            # forward
            predictions = model(inputs)

            # calculate loss
            batch_loss = F.nll_loss(predictions, targets)
            epoch_loss += batch_loss.item()

            # backward propagation
            batch_loss.backward()

            # update weights
            optimizer.step()
        avg_train_loss = epoch_loss / len(train_data.dataset)
        train_acc, _ = evaluate_performance(model, train_data, tag_decoder)

        # eval
        eval_acc, eval_loss = evaluate_performance(model, eval_data, tag_decoder)

        eval_accuracy_history.append(eval_acc)
        eval_loss_history.append(eval_loss)

        print(f"Epoch {epoch}: train_loss={avg_train_loss:.6f}, train_acc={train_acc:.4f}, "
              f"eval_loss={eval_loss:.6f}, eval_acc={eval_acc:.4f}")
    return eval_loss_history, eval_accuracy_history


def evaluate_performance(model, data_loader, tag_decoder):
    model.eval()

    # init
    correct = 0.0
    total = 0.0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in data_loader:
            # forward
            outputs = model(inputs)

            # calculate loss
            batch_loss = F.nll_loss(outputs, targets)
            total_loss += batch_loss.item()

            _, predicted = torch.max(outputs.data, 1)
            if IS_ENTITY_RECOGNITION:
                for pred, true_tag in zip(predicted.numpy(), targets.numpy()):
                    total += 1
                    if pred == true_tag:
                        # if both are 'O', don't count
                        # True Negative
                        if tag_decoder[int(pred)] == 'O':
                            total -= 1
                        else:
                            # False Negative
                            correct += 1
            else:
                matches = (predicted == targets)
                total += targets.size(0)
                correct += matches.sum().item()
    return correct / total, total_loss / len(data_loader.dataset)


def load_corpus(filename, add_markers=True):
    sentences = []
    labels = []
    with open(filename, "r", encoding="utf-8") as f:
        content = f.readlines()
        current_sentence = []
        current_labels = []
        for line in content:
            line = line.strip()
            if line:
                parts = line.split(FIELD_SEPARATOR)
                word = parts[0]
                current_sentence.append(word)
                if len(parts) > 1:
                    tag = parts[1]
                    current_labels.append(tag)
            else:
                if current_sentence:
                    if add_markers:
                        current_sentence = ['<s>', '<s>'] + current_sentence + ['<e>', '<e>']
                    sentences.append(current_sentence)
                    if current_labels:
                        labels.append(current_labels)
                # init for next sentence
                current_sentence = []
                current_labels = []
    return sentences, labels


def load_test_data(filename):
    processed_sentences = []
    original_sentences = []
    with open(filename, "r", encoding="utf-8") as f:
        content = f.readlines()
        current_sentence = []
        for line in content:
            line = line.strip()
            if line:
                word = line.split()[0]
                current_sentence.append(word)
            else:
                if current_sentence:
                    original_sentences.append(current_sentence.copy())
                    marked_sentence = ['<s>', '<s>'] + current_sentence + ['<e>', '<e>']
                    processed_sentences.append(marked_sentence)
                current_sentence = []
    return processed_sentences, original_sentences


def handle_rare_words(sentences, threshold=1, unk_token='<unknown>'):
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(sentence)
    rare_words = {word for word, count in word_counts.items() if count <= threshold}
    for sentence in sentences:
        for i in range(len(sentence)):
            if sentence[i] in rare_words:
                sentence[i] = unk_token
    return sentences


def build_vocabularies(word_data, tag_data, unk_token='<unknown>'):
    all_words = []
    for sentence in word_data:
        all_words.extend(sentence)
    all_words.append(unk_token)
    unique_words = sorted(set(all_words))
    word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
    idx_to_word = {idx: word for idx, word in enumerate(unique_words)}
    all_tags = []
    for sentence_tags in tag_data:
        all_tags.extend(sentence_tags)
    unique_tags = sorted(set(all_tags))
    tag_to_idx = {tag: idx for idx, tag in enumerate(unique_tags)}
    idx_to_tag = {idx: tag for idx, tag in enumerate(unique_tags)}
    return word_to_idx, idx_to_word, tag_to_idx, idx_to_tag


def convert_to_indices(sentences, vocab, unk_token='<unknown>'):
    indexed_sentences = []
    for sentence in sentences:
        indexed_sentence = []
        for word in sentence:
            index = vocab.get(word, vocab.get(unk_token))
            indexed_sentence.append(index)
        indexed_sentences.append(indexed_sentence)
    return indexed_sentences


def create_context_windows(data, labels=None, window_size=5):
    half_window = window_size // 2
    windows = []
    targets = []
    if labels is not None:
        for sentence, sentence_labels in zip(data, labels):
            for i in range(half_window, len(sentence) - half_window):
                context = [sentence[i + j] for j in range(-half_window, half_window + 1)]
                windows.append(context)
                label_idx = i - half_window
                if 0 <= label_idx < len(sentence_labels):
                    targets.append(sentence_labels[label_idx])
        return np.array(windows), np.array(targets)
    else:
        sentence_windows = []
        for sentence in data:
            current_windows = []
            for i in range(half_window, len(sentence) - half_window):
                context = [sentence[i + j] for j in range(-half_window, half_window + 1)]
                current_windows.append(context)
            sentence_windows.append(current_windows)
        return sentence_windows


def create_visualization(output_dir, data, filename, title_text):
    fig = plt.figure(figsize=(8, 6))
    epochs = range(len(data))
    plt.plot(epochs, data, linewidth=2.0)
    plt.xlabel("Epochs")
    plt.ylabel(title_text)
    plt.title(f"{title_text} vs Epochs")
    # Save the figure
    plt.savefig(f"{output_dir}/{filename}.png", dpi=192)
    plt.close(fig)


def generate_predictions(model, processed_data, original_data, tag_decoder):
    output_filename = f"./test1.{TASK_TYPE}"

    if os.path.exists(output_filename):
        os.remove(output_filename)

    with open(output_filename, "w", encoding="utf-8") as f:
        with torch.no_grad():
            for sentence_windows, original_sentence in zip(processed_data, original_data):
                input_tensor = torch.LongTensor(sentence_windows)
                outputs = model(input_tensor)
                predicted_indices = torch.argmax(outputs, dim=1).numpy()
                for word, pred_idx in zip(original_sentence, predicted_indices):
                    tag = tag_decoder[pred_idx]
                    f.write(f"{word} {tag}\n")
                f.write("\n")


def main():
    output_dir = f"{TASK_TYPE}_results"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading training data...")
    train_sentences, train_tags = load_corpus(f"./{TASK_TYPE}/train")

    train_sentences = handle_rare_words(train_sentences)
    word_to_idx, idx_to_word, tag_to_idx, idx_to_tag = build_vocabularies(train_sentences, train_tags)
    train_indices = convert_to_indices(train_sentences, word_to_idx)
    train_tag_indices = convert_to_indices(train_tags, tag_to_idx)
    train_windows, train_targets = create_context_windows(train_indices, train_tag_indices)
    train_dataset = TensorDataset(torch.LongTensor(train_windows), torch.LongTensor(train_targets))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Loading validation data...")
    dev_sentences, dev_tags = load_corpus(f"./{TASK_TYPE}/dev")
    dev_indices = convert_to_indices(dev_sentences, word_to_idx)
    dev_tag_indices = convert_to_indices(dev_tags, tag_to_idx)
    dev_windows, dev_targets = create_context_windows(dev_indices, dev_tag_indices)
    dev_dataset = TensorDataset(torch.LongTensor(dev_windows), torch.LongTensor(dev_targets))
    dev_loader = DataLoader(dev_dataset, batch_size=VALIDATION_BATCH_SIZE, shuffle=True)

    print("Loading test data...")
    test_sentences, original_test = load_test_data(f"./{TASK_TYPE}/test")
    test_indices = convert_to_indices(test_sentences, word_to_idx)
    test_windows = create_context_windows(test_indices)

    print("Initializing model...")
    model = Mlp(
        vocabulary_size=len(word_to_idx),
        hidden_size=128,
        num_classes=len(tag_to_idx))
    optimizer = optim.Adam(model.parameters())

    print(f"Training model for {TRAINING_EPOCHS} epochs...")
    dev_loss_history, dev_accuracy_history = train_model(
        model, optimizer, train_loader, dev_loader,
        TRAINING_EPOCHS, idx_to_tag
    )

    print("Creating visualizations...")
    create_visualization(output_dir, dev_loss_history, "validation_loss", "Validation Loss")
    create_visualization(output_dir, dev_accuracy_history, "validation_accuracy", "Validation Accuracy")


    print("Generating predictions...")
    generate_predictions(model, test_windows, original_test, idx_to_tag)


if __name__ == "__main__":
    main()