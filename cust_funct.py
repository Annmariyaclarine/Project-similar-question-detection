import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import re
import pickle
import keras

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def loss(margin=1):
    def contrastive_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss


# Cleaning the questions
def Cleaning(text):
    corpus = []
    wnl = WordNetLemmatizer()

    for q_string in list(text):
        # Cleaning special character from the question
        quest = re.sub(pattern='[^a-zA-Z]', repl=' ', string=q_string)
        # Converting the entire question into lower case
        quest = quest.lower()
        # to remove numeric digits from string
        quest = ''.join([i for i in quest if not i.isdigit()])
        # Tokenizing the question by words
        words = quest.split()
        # Removing the stop words
        filtered_words = [word for word in words if word not in set(stopwords.words('english'))]
        # Lemmatizing the words
        lemmatized_words = [wnl.lemmatize(word) for word in filtered_words]
        # Joining the lemmatized words
        quest = ' '.join(lemmatized_words)
        # Building a corpus of quest
        corpus.append(quest)
    return corpus


# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
maxlen = 100
model = load_model('1conlossbimalstm.h5', custom_objects={'loss': loss, 'K': keras.backend})

def pred(question1, question2):
    cq1 = Cleaning([question1])
    cq2 = Cleaning([question2])
    tok_q1 = tokenizer.texts_to_sequences(cq1)
    tok_q2 = tokenizer.texts_to_sequences(cq2)
    pad_q1 = pad_sequences(tok_q1, maxlen=maxlen, padding='post')
    pad_q2 = pad_sequences(tok_q2, maxlen=maxlen, padding='post')
    similarity_prob = model.predict([pad_q1, pad_q2])
    threshold = 0.37
    if similarity_prob >= threshold:
        similarity_label = "similar"
    else:
        similarity_label = "not similar"
    return similarity_label,similarity_prob
    # return cq1, cq2, tok_q1, tok_q2, pad_q1, pad_q2,similarity_prob,similarity_label
    # return similarity_prob, similarity_label
