
import os
import re
import magic
import librosa
from treinamento import audio
import numpy as np
#from keras.utils import np_utils
from sklearn.feature_extraction.text import CountVectorizer
from joblib import Parallel, delayed
from treinamento import audio
import time
import psutil


'''
REFERÊNCIAS: 

https://panda.ime.usp.br/pensepy/static/pensepy/10-Arquivos/files.html
https://github.com/aravindpai/Speech-Recognition/blob/master/Speech%20Recognition.ipynb
https://medium.com/@patrickbfuller/librosa-a-python-audio-libary-60014eeaccfb
https://keras.io/examples/audio/speaker_recognition_using_cnn/
https://blogs.rstudio.com/ai/posts/2018-06-06-simple-audio-classification-keras/
https://github.com/manashmandal/DeadSimpleSpeechRecognizer/blob/master/preprocess.py
https://medium.com/manash-en-blog/building-a-dead-simple-word-recognition-engine-using-convnet-in-keras-25e72c19c12b
https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial
https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

Exploiting spectro-temporal locality in deep learning based acoustic event detection
https://link.springer.com/article/10.1186/s13636-015-0069-2

https://python-speech-features.readthedocs.io/en/latest/
https://speechpy.readthedocs.io/_/downloads/en/stable/pdf/

https://librosa.org/doc/latest/feature.html
https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html
https://stackoverflow.com/questions/60492462/mfcc-python-completely-different-result-from-librosa-vs-python-speech-features
https://github.com/astorfi/speechpy/blob/master/speechpy/feature.py
https://datascience.stackexchange.com/questions/27634/how-to-convert-a-mel-spectrogram-to-log-scaled-mel-spectrogram



'''
class PreProcessamento(object):


    listaGlobalAudios = []
    vocabulario = []


    def __init__(self, configuracao):
        # https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
        self.configuracao = configuracao



    def carregarListaAudiosNomesArquivosTranscricoes(self):

        print('Iniciando montagem da lista com nomes dos arquivos de áudio e transcrições')

        caminho_arquivos_treinamento = '../../corpus'

        for (root, dirs, arquivos) in os.walk(caminho_arquivos_treinamento):
            for arquivo in arquivos:
                if '.data' in arquivo and '.data.orig' not in arquivo:

                    arquivo_utts = os.path.join(root, arquivo)

                    '''
                    https://stackoverflow.com/questions/436220/how-to-determine-the-encoding-of-text
                    UnicodeDecodeError: 'utf-8' codec can't decode byte 0x92 in position 16: invalid start byte
                    Tentando evitar UnicodeDecodeError
                    '''
                    blob = open(arquivo_utts, 'rb').read()
                    m = magic.Magic(mime_encoding=True)
                    encoding = m.from_buffer(blob)

                    # https://stackoverflow.com/questions/16465399/need-the-path-for-particular-files-using-os-walk
                    ref_arquivo = open(arquivo_utts, "r", encoding=encoding)
                    padrao_regex = '(\d\d\d_yoruba_.*_headset_)(\d\d\d)(\d)?'


                    try:

                       for linha in ref_arquivo:

                            nome_arquivo_audio = re.search(padrao_regex, linha).group()
                            transcricao = re.search('".*"', linha).group().replace('"', '')
                            audioObj = audio.Audio(nome_arquivo_audio, None, transcricao, None, None)
                            self.listaGlobalAudios.append(audioObj)
                            self.vocabulario.append(transcricao)

                    except UnicodeError as e:
                        pass

                    ref_arquivo.close()




    def montarListaCaminhosArquivosAudio(self):

        print('Iniciando atualização de caminhos para arquivos de áudio')
        caminho_arquivos_treinamento = '../../corpus'


        for audio in self.listaGlobalAudios:
            for (root, dirs, arquivos) in os.walk(caminho_arquivos_treinamento):
                for arquivo in arquivos:
                    if audio.nome_arquivo + '.wav' in arquivo:
                        caminho_audio = os.path.join(root, audio.nome_arquivo + '.wav')
                        audio.caminho_arquivo = caminho_audio



    def carregarListaGlobalAudiosTreinamento(self):

        print('Iniciando conversão dos audios em espectogramas e log_energy')

        parallel = Parallel(n_jobs=self.configuracao['n_jobs'],
                            backend=self.configuracao['backend'],
                            verbose=self.configuracao['verbose'])

        parallel(delayed(self.extrairLogEnergyMelSpectogram_Paralelizado)(audio) for audio in self.listaGlobalAudios)


    def extrairLogEnergyMelSpectogram_Paralelizado(self, audio):

        dimensao_maxima = 50
        sinal_audio, sample_rate = librosa.load(audio.caminho_arquivo, sr=16000)
        espectograma = librosa.feature.melspectrogram(y=sinal_audio, sr=sample_rate)

        '''
                        https://en.wikipedia.org/wiki/Mel-frequency_cepstrum

                        The experimental results in Section V show
                        a consistent improvement in overall system performance by
                        using the log-energy feature. There has been some question
                        as to whether this improvement holds in larger-scale ASR
                        tasks [40]. Nevertheless, these experiments at least show that
                        nothing in principle prevents frequency-independent features
                        such as log-energy from being accommodated within a CNN
                        architecture when they stand to improve performance. (p.1539)   
                        https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/CNN_ASLPTrans2-14.pdf?irgwc=1&OCID=AID2000142_aff_7806_1246483&tduid=%28ir__3n1mp6niookftmjtkk0sohzjxm2xilegkfdgcd0u00%29%287806%29%281246483%29%28%283283fab34b8e9cb7166fb504c2f02716%29%2881561%29%28686431%29%28at106140_a107739_m12_p12460_cBR%29%28%29%29%283283fab34b8e9cb7166fb504c2f02716%29&irclickid=_3n1mp6niookftmjtkk0sohzjxm2xilegkfdgcd0u00 

                        https://stackoverflow.com/questions/60492462/mfcc-python-completely-different-result-from-librosa-vs-python-speech-features                    

                        https://pytorch.org/audio/transforms.html#spectrogram

                        https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html

                        http://man.hubwiz.com/docset/LibROSA.docset/Contents/Resources/Documents/_modules/librosa/core/spectrum.html#power_to_db


                        def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
                        Convert a power spectrogram (amplitude squared) to decibel (dB) units

                        This computes the scaling ``10 * log10(S / ref)`` in a numerically
                        stable way.

                        TECHNIQUES FOR FEATURE EXTRACTION IN SPEECH
                        RECOGNITION SYSTEM : A COMPARATIVE STUDY 
                        https://arxiv.org/pdf/1305.1145.pdf

                        https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
                        https://en.wikipedia.org/wiki/Spectral_density#Power_spectral_density

                        Não estou usando DCT, mas log-energy 
                        https://en.wikipedia.org/wiki/Discrete_cosine_transform

                        https://docs.python.org/2/tutorial/datastructures.html#dictionaries
                        https://developer.rhino3d.com/guides/rhinopython/primer-101/6-tuples-lists-dictionaries/

        '''

        log_energy_espectograma = librosa.power_to_db(espectograma)

        '''
        Garante que os vetores mfcc tenham o mesmo tamanho fixo através de um padding de 0 
        em volta do vetor mfcc, caso ele tenha dimensão menor do que um valor máximo pré-fixado

        https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial

        How to normalize MFCCs
        https://www.kaggle.com/c/freesound-audio-tagging/discussion/54082

        '''

        if (dimensao_maxima > log_energy_espectograma.shape[1]):

            padding = dimensao_maxima - log_energy_espectograma.shape[1]
            mel_frequency_cepstrum_coefficients = np.pad(log_energy_espectograma, pad_width=((0, 0), (0, padding)),
                                                         mode='constant')

        # Else cutoff the remaining parts
        else:
            mel_frequency_cepstrum_coefficients = log_energy_espectograma[:, : dimensao_maxima]

        audio.log_energy = log_energy_espectograma


    vetorizador = CountVectorizer()


    def converterTranscricaoCategoricalDecoder(self):
        '''

        from keras.utils import to_categorical

        https://keras.io/examples/
        https://keras.io/examples/audio/speaker_recognition_using_cnn/
        https://github.com/attardi/CNN_sentence
        https://github.com/attardi/CNN_sentence/blob/master/process_data.py
        https://scikit-learn.org/stable/modules/multiclass.html
        https://medium.com/@maobedkova/acoustic-word-embeddings-fc3f1a8f0519
        https://medium.com/@oyewusiwuraola/yor%C3%B9b%C3%A1-word-vector-representation-with-fasttext-fe905bf558ea
        https://github.com/Niger-Volta-LTI

        https://www.youtube.com/channel/UCoEHw2cfZ0YJNQUKeWxLuWg

        '''

        # labels_encoded[0].vocabulary_   devolve o índice de cada palavra
        #labels_encoded = self.vetorizador(dicionario_treinamento_raw.values())
        #print(labels_encoded[0].inverse_transform(labels_encoded[1]))

        inicio_vetorizacao = time.clock()

        self.vetorizador.fit(self.vocabulario)

        parallel = Parallel(n_jobs=self.configuracao['n_jobs'],
                            backend=self.configuracao['backend'],
                            verbose=self.configuracao['verbose'])

        parallel(delayed(self.vetorizar_transcricao)(audio) for audio in self.listaGlobalAudios)


        processamento_vetorizacao = time.clock() - inicio_vetorizacao
        print('Tempo de processamento da vetorizacao {}'.format(processamento_vetorizacao))

        '''
        Testando voltar para a transcrição original após vetorização
        Preciso garantir aqui que as transcrições vetorizadas combinem exatamente com os audios
        '''
        for audio in self.listaGlobalAudios:
            transcricao_convertida_teste = self.vetorizador.inverse_transform(audio.label_encoded)
            print(transcricao_convertida_teste)
            break


    def vetorizar_transcricao(self, audio):
        vetor_encoded = self.vetorizador.transform([audio.transcricao])
        audio.label_encoded = vetor_encoded


    def vetorizador_sequence(self, vetorizador, listaSentencas):

        '''

        https://www.oreilly.com/library/view/applied-text-analysis/9781491963036/ch04.html
        https://developers.google.com/machine-learning/guides/text-classification/step-3?hl=pl


        https://developers.google.com/machine-learning/guides/text-classification/step-4?hl=pl

        DECISÕES:

        1. Precisarei usar sequence models e word embedding
        2. Usar um word embedding já treinado ou treinar com os dados que possuo ?
        3. Usarei AWE (Acoustic Word Embedding) ou somente TWE (Textual Word Embedding) ?

        LINKS ÚTEIS:
        https://medium.com/@maobedkova/acoustic-word-embeddings-fc3f1a8f0519
        https://medium.com/@oyewusiwuraola/yor%C3%B9b%C3%A1-word-vector-representation-with-fasttext-fe905bf558ea
        https://github.com/Niger-Volta-LTI


        https://fasttext.cc/docs/en/pretrained-vectors.html
        P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information

        @article{bojanowski2017enriching,
        title={Enriching Word Vectors with Subword Information},
        author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
        journal={Transactions of the Association for Computational Linguistics},
        volume={5},
        year={2017},
        issn={2307-387X},
        pages={135--146}
        }

        wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.yo.zip
        Length: (2,1 GB)

        '''
        pass



    def obterDados(self):

        inicio = time.clock()

        self.carregarListaAudiosNomesArquivosTranscricoes()
        self.converterTranscricaoCategoricalDecoder()
        self.vocabulario = [] # Neste ponto não preciso mais da lista de vocabulários
        self.montarListaCaminhosArquivosAudio()
        self.carregarListaGlobalAudiosTreinamento()

        '''
        !!! StackOverflow !!!
        parallel = Parallel(backend="threading", verbose=1)
        parallel(delayed(treinamento.carregarListaGlobalAudiosTreinamento)(key, treinamento.dicionario_treinamento_encoded[key]) for key in treinamento.dicionario_treinamento_encoded)
        '''

        tempo_processamento = time.clock() - inicio

        print(self.listaGlobalAudios[0])

        print('Tempo total de pré-processamento dos dados:  {} segundos'.format(tempo_processamento))

        return self.listaGlobalAudios







if __name__ == '__main__':

    '''
    Importei o arquivo para monitoramento de memória do sequinte projeto: 
    https://github.com/astrofrog/psrecord/blob/master/psrecord/main.py
    
    Arquivo importado no diretório: /usr/lib/python3.6/
    Nome: monitoramento_memoria.py 
    
    '''
    from monitoramento import monitoramento_PROPRIETARY as monitoramento_memoria
    #from monitoramento import monitoramento_memoria

    import datetime

    pid_python = os.getpid()

    path = '/home/usuario/mestrado/yorubaSpeechRecognition/monitoramento'
    arquivoLog = os.path.join(path, f'yorubaSpeechRecognition__'
                                    f'{str(datetime.datetime.today())}__'
                                    f'{str(datetime.time())}__')

    from multiprocessing import Process
    '''
    https://stackabuse.com/parallel-processing-in-python/
    https://www.machinelearningplus.com/python/parallel-processing-python/#:~:text=In%20python%2C%20the%20multiprocessing%20module,in%20completely%20separate%20memory%20locations.
    
    
    https://psutil.readthedocs.io/en/release-2.2.1/
    '''

    '''
    PARÂMETROS DE PARALELIZAÇÃO
    https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
    
    n_jobs = 4   # máximo número de cpus = psutil.cpu_count() 
    
    using ‘n_jobs=1’ enables to turn off parallel computing for debugging without changing the codepath

    backend = "multiprocessing"    
    backend = "threading"
    
    “loky” used by default, can induce some communication and memory overhead when exchanging input and output data with the worker Python processes.
“multiprocessing” previous process-based backend based on multiprocessing.Pool. Less robust than loky.
“threading” is a very low-overhead backend but it suffers from the Python Global Interpreter Lock if the called function relies a lot on Python objects. “threading” is mostly useful when the execution bottleneck is a compiled extension that explicitly releases the GIL (for instance a Cython loop wrapped in a “with nogil” block or an expensive call to a library such as NumPy).
finally, you can register backends by calling register_parallel_backend. This will allow you to implement a backend of your liking.

    
    
    
    TESTAR DIFERENTES PARÂMETROS DE PARALELIZAÇÃO E VER O EFEITO EM TEMPO E MEMÓRIA
    '''


    configuracao_paralelizacao = {}
    configuracao_paralelizacao['n_jobs'] = 4
    configuracao_paralelizacao['verbose'] = 5
    backend = ["loky", "multiprocessing", "threading"]
    configuracao_paralelizacao['backend'] = backend[1]

    preProcessamento = PreProcessamento(configuracao_paralelizacao)

    '''
    PRÉ-PROCESSAMENTO
    '''
    p1 = Process(target=preProcessamento.obterDados)
    p1.start()


    '''
    Processo rodando em paralelo para monitora uso de memória, CPU, tempo decorrido no pré-processamento    
    '''
    p2 = Process(target=monitoramento_memoria.monitor, args=(configuracao_paralelizacao, p1.pid, arquivoLog + '.txt', arquivoLog + '.png'))
    p2.start()
