

def load_dataset(input_path, output_path):
    """ Loads a dataset given specified input and output path """
    input_lines = open(input_path).readlines()
    output_lines = open(output_path).readlines()

    assert(len(input_lines) == len(output_lines))
    in_set = generate_vocab_iterations(input_lines)
    out_set = generate_vocab_sentences(output_lines)
    return {'input' : in_set, 'output' : out_set}

def generate_vocab_iterations(train_lines):
    """ Generates vocab indeces and converts sentences to integer values """

    word_to_idx = {}
    idx_to_word = {}
    curr_index = 1

    # First pass: generate vocab
    for line in lines:
        tokens = line.lower().replace(".", " <end>").split(" ")
        for token in tokens:
            word_to_idx[token] = curr_index
            idx_to_word[curr_index] = token
            curr_index += 1

    sentences = []
    # Second pass: generate vocab
    for line in lines:
        sentence = []
        tokens = line.lower().replace(".", " <end>").split(" ")
        for token in tokens:
            sentence.append(word_to_idx[tokens])

        sentences.append(sentence)

    return {'sentences' : sentences, 'word_to_idx' : word_to_idx, 'idx_to_word' : idx_to_word}
