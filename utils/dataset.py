from nltk.corpus import movie_reviews
import nltk

def generateDataset():
    documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
    words = movie_reviews.words()

    all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
    word_to_idx = { "<unk>" : 0}
    idx_to_word = { 0 : "<unk>" }

    curr_index = 1
    for word in words:
        numOccurences = all_words[word.lower()]
        if numOccurences > 10:
            word_to_idx[word] = curr_index
            idx_to_word[curr_index] = word.lower()
            curr_index += 1
    random.shuffle(documents)