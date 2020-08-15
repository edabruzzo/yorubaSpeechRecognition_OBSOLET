
import os
import argparse
import time
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
#from keras import layers
#from keras import layers
#import keras
#from keras.models import Sequential

import warnings
warnings.filterwarnings('ignore')

from treinamento.preprocessamento import PreProcessamento
from util.paralelizacao import Paralelizacao
from util.sequencial import Sequencial
from util.monitoramento_memoria import Monitoramento

"""


Module to train sequence model with fine-tuned pre-trained embeddings.
Vectorizes training and validation texts into sequences and uses that for
training a sequence model - a sepCNN model. We use sequence model with
pre-trained embeddings that are fine-tuned for text classification when the
ratio of number of samples to number of words per sample for the given dataset
is neither small nor very large (~> 1500 && ~< 15K).



https://github.com/google/eng-edu/blob/master/ml/guides/text_classification/train_fine_tuned_sequence_model.py

https://developers.google.com/machine-learning/guides/text-classification/conclusion?hl=pl


https://towardsdatascience.com/customer-case-study-building-an-end-to-end-speech-recognition-model-in-pytorch-with-assemblyai-473030e47c7c

https://github.com/attardi/CNN_sentence/blob/master/process_data.py

https://scikit-learn.org/stable/modules/multiclass.html

https://en.wikipedia.org/wiki/Multi-label_classification

https://keras.io/examples/nlp/text_classification_from_scratch/#two-options-to-vectorize-the-data

https://www.tensorflow.org/tutorials/text/word_embeddings#encode_each_word_with_a_unique_number

https://github.com/manashmandal/DeadSimpleSpeechRecognizer

https://en.wikipedia.org/wiki/Discrete_cosine_transform

https://en.wikipedia.org/wiki/Spectral_density#Power_spectral_density

https://en.wikipedia.org/wiki/Mel-frequency_cepstrum

https://keras.io/examples/audio/speaker_recognition_using_cnn/

https://github.com/aravindpai/Speech-Recognition/blob/master/Speech%20Recognition.ipynb

https://medium.com/manash-en-blog/building-a-dead-simple-word-recognition-engine-using-convnet-in-keras-25e72c19c12b

https://medium.com/@oyewusiwuraola/yor%C3%B9b%C3%A1-word-vector-representation-with-fasttext-fe905bf558ea

https://fasttext.cc/docs/en/pretrained-vectors.html


"""




class TreinaModelo(object):

    embedding_data_dir = '../../embeddings_yoruba'

    # Limit on the number of features. We use the top 20K features.
    TOP_K = 20000

    # Range (inclusive) of n-gram sizes for tokenizing text.
    NGRAM_RANGE = (1, 2)

    # Whether text should be split into word or character n-grams.
    # One of 'word', 'char'.
    TOKEN_MODE = 'word'

    # Limit on the length of text sequences. Sequences longer than this
    # will be truncated.
    MAX_SEQUENCE_LENGTH = 500


    def _get_embedding_matrix(self, word_index):
        """

        # References:

            https://fasttext.cc/docs/en/pretrained-vectors.html
            Download and uncompress archive from:
            https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.yo.zip

            https://github.com/google/eng-edu/blob/master/ml/guides/text_classification/load_data.py
            https://developers.google.com/machine-learning/guides/text-classification/conclusion?hl=pl
        """

        # Read the pre-trained embedding file and get word to word vector mappings.
        embedding_matrix_all = {}

        fname = os.path.join(self.embedding_data_dir, 'wiki.yo.vec')

        with open(fname) as f:
            for line in f:  # Every line contains word followed by the vector value
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                #Montando um dicionário de embedding words com coeficientes
                embedding_matrix_all[word] = coefs

        # Prepare embedding matrix with just the words in our word_index dictionary
        num_words = min(len(word_index) + 1, self.TOP_K)
        embedding_matrix = np.zeros((num_words, embedding_matrix_all.shape))

        for word, i in word_index.items():
            if i >= self.TOP_K:
                continue
            embedding_vector = embedding_matrix_all.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix


    def train_fine_tuned_sequence_model(self, data,
                                    embedding_data_dir,
                                    learning_rate=1e-3,
                                    epochs=1000,
                                    batch_size=128,
                                    blocks=2,
                                    filters=64,
                                    dropout_rate=0.2,
                                    embedding_dim=200,
                                    kernel_size=3,
                                    pool_size=3):

        # Get the data.
        (train_audios, train_labels), (val_audios, val_labels) = data

        # Vectorize texts.
        y_train, y_val, word_index = self.sequence_vectorize(train_labels, val_labels)

        # Number of features will be the embedding input dimension. Add 1 for the
        # reserved index 0.
        #num_features = min(len(word_index) + 1, self.TOP_K)

        embedding_matrix = self._get_embedding_matrix()

        loss = 'sparse_categorical_crossentropy'
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

        # Create callback for early stopping on validation loss. If the loss does
        # not decrease in two consecutive tries, stop training.
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=2)]


        '''
        
        Aqui surge uma questão interessante: 
        
        Text Word Embeddings estão sendo usadas no modelo como se fossem features e não são !!!
        As word embeddings fazem parte dos labels a serem preditos e não das features 
        
        Nesse momento surge a discussão sobre a necessidade delas e se elas vão atrapalhar o modelo
        Além disso, surge a questão sobre Acoustic Word Embeddings
        
        "Acoustic Word Embedding (AWE) is a fixed-dimensional representation of a variable-length audio signal in an embedding space. The idea of AWEs is quite close to well-known textual word embeddings which create similar vector representations for semantically similar words. However, AWEs aim to model acoustic similarity rather than semantic similarity.
        https://medium.com/@maobedkova/acoustic-word-embeddings-fc3f1a8f0519
        
        https://static.googleusercontent.com/media/research.google.com/pt-BR//pubs/archive/42543.pdf
        Word Embeddings for Speech Recognition
        Samy Bengio and Georg Heigold

        https://www.oreilly.com/library/view/applied-text-analysis/9781491963036/ch04.html

  
        '''
        # layer and let it fine-tune to the given dataset.
        model = self.sepcnn_model(blocks=blocks,
                                     filters=filters,
                                     kernel_size=kernel_size,
                                     embedding_dim=embedding_dim,
                                     dropout_rate=dropout_rate,
                                     pool_size=pool_size,
                                     input_shape=train_audios.shape[1:],
                                     num_classes=len(word_index),
                                     #num_features=num_features,
                                     use_pretrained_embedding=True,
                                     is_embedding_trainable=True,
                                     embedding_matrix=embedding_matrix)

        # Compile model with learning parameters.
        model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

        # Load the weights that we had saved into this new model.
        model.load_weights('sequence_model_with_pre_trained_embedding.h5')

        # Train and validate model.
        history = model.fit(train_audios,
                        train_labels,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(val_audios, val_labels),
                        verbose=2,  # Logs once per epoch.
                        batch_size=batch_size)

        # Print results.
        history = history.history
        print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

        # Save model.
        model.save('tweet_weather_sepcnn_fine_tuned_model.h5')
        return history['val_acc'][-1], history['val_loss'][-1]


    def sequence_vectorize(self, train_labels, val_labels):

        # Create vocabulary with training texts.
        tokenizer = text.Tokenizer(num_words=self.TOP_K)
        tokenizer.fit_on_texts(train_labels)

        # Vectorize training and validation texts.
        y_train = tokenizer.texts_to_sequences(train_labels)
        y_val = tokenizer.texts_to_sequences(val_labels)

        # Get max sequence length.
        max_length = len(max(y_train, key=len))
        if max_length > self.MAX_SEQUENCE_LENGTH:
            max_length = self.MAX_SEQUENCE_LENGTH

        # Fix sequence length to max value. Sequences shorter than the length are
        # padded in the beginning and sequences longer are truncated
        # at the beginning.
        y_train = sequence.pad_sequences(y_train, maxlen=max_length)
        y_val = sequence.pad_sequences(y_val, maxlen=max_length)

        return y_train, y_val, tokenizer.word_index


    def obter_conjuntos_treinamento_validacao(self):
        '''

        https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html

        :return:

        '''

        path = '/home/usuario/mestrado/yorubaSpeechRecognition/dadosVetorizados/audios_vetorizados'
        data = pd.read_csv(os.path.join(path, 'dataset.csv'))
        vocabulario = data[:, -1]
        vetorizador = CountVectorizer()
        y = vetorizador.fit_transform(vocabulario)  # Scaling the Feature columns
        scaler = StandardScaler()
        X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype=float))

        # https://github.com/aravindpai/Speech-Recognition/blob/master/Speech%20Recognition.ipynb
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            stratify=vocabulario,
                                                            test_size=0.2,
                                                            random_state=777,
                                                            shuffle=True)

        return ((X_train, y_train), (X_test, y_test))




if __name__ == '__main__':

    '''
    
    https://www.tensorflow.org/tutorials
    
    '''

    inicio = time.clock()

    treina = TreinaModelo()

    '''
    #USADO PARA TESTAR DIFERENTES CONFIGURAÇÕES DE PARALELISMO
    backend = ["loky", "multiprocessing", "threading"]
    preprocessamento.PreProcessamento(numJobs=4, backend=backend[1], verbose=5)\
                    .carregarListaGlobalAudiosTreinamento_(paralelo=False, monitorarExecucao=False)
    
    
    #Configuração padrão: n_jobs=4, backend=multiprocessing, verbose=5, paralelo=True, monitorarExecucao=True
    preprocessamento.PreProcessamento()\
                    .carregarListaGlobalAudiosTreinamento_(paralelo=True, monitorarExecucao=False)

    listaTreinamento = preprocessamento.PreProcessamento().listaGlobalAudios
    print(len(listaTreinamento))
    
    '''

    '''
    path = '/home/usuario/mestrado/yorubaSpeechRecognition/treinamento/dadosVetorizados/'
    path = '/home/usuario/mestrado/yorubaSpeechRecognition/treinamento/dadosVetorizados'
    '''

    processa = PreProcessamento(executarEmParalelo=True)
    Monitoramento().monitorar_memoria(processa.obterDados, configuracao_paralelizacao=processa.configuracao_paralelismo)

    (X_train, y_train), (X_test, y_test) = treina.obter_conjuntos_treinamento_validacao( )

    print(X_train.shape)
    print(y_train.shape)

    tempo_treinamento_modelo = time.clock() - inicio
    print('Tempo total de treinamento do modelo:  {} segundos'.format(tempo_treinamento_modelo))
