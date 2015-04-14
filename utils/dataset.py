

def load_dataset(input_path, output_path):
    """ Loads a dataset given specified input and output path """
    input_lines = open(input_path).readlines()
    output_lines = open(output_path).readlines()

    assert(len(input_lines) == len(output_lines))
    in_set = generate_vocab_iterations(input_lines)
    out_set = generate_vocab_iterations(output_lines)
    return {'input' : in_set, 'output' : out_set}

def generate_vocab_iterations(lines):
    """ Generates vocab indeces and converts sentences to integer values """

    word_to_idx = {'<START>' : 0, '<END>' : 1}
    idx_to_word = {0: '<START>', 1 : '<END>'}
    curr_index = 2

    # First pass: generate vocab
    for line in lines:
        tokens = tokenize_line(line)
        for token in tokens:
            if word_to_idx.has_key(token):
                continue

            word_to_idx[token] = curr_index
            idx_to_word[curr_index] = token
            curr_index += 1

    sentences = []
    # Second pass: generate vocab
    for line in lines:
        sentence = []
        tokens = tokenize_line(line)
        for token in tokens:
            sentence.append(word_to_idx[token])

        sentences.append(sentence)

    return {'sentences' : sentences, 'word_to_idx' : word_to_idx, 'idx_to_word' : idx_to_word, 
            'size' : len(word_to_idx)}

def get_sentence(Xi, idx_to_word):
    " Converts an array of indeces in an array to a word list, returns a sentence "
    words = []
    for x in Xi:
        words.append(idx_to_word[x])

    return ' '.join(words)

def tokenize_line(line, add_terminations = True):
    """ Tokenizes a line, converts to lowercase, adds <START>, <END> tokens
        @returns: an array of tokens of the line
        @
    """
    if add_terminations:
        tokens = ['<START>']

    tokens.extend(line.lower().replace(".", " .").replace("\n", "").split(" "))

    if add_terminations:
        tokens.append('<END>')

    return tokens