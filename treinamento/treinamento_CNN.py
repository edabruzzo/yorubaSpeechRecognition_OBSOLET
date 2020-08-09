
import os
import re
import magic
import librosa
from treinamento import audio

'''
REFERÊNCIAS: 

https://panda.ime.usp.br/pensepy/static/pensepy/10-Arquivos/files.html
https://github.com/aravindpai/Speech-Recognition/blob/master/Speech%20Recognition.ipynb
https://medium.com/@patrickbfuller/librosa-a-python-audio-libary-60014eeaccfb

'''
class TreinamentoCNN(object):


    def obterDicionarioTreinamento(self):

        caminho_arquivos_treinamento = '../../corpus'
        treinamento_dicionario = {}

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

                            arquivo_audio = re.search(padrao_regex, linha).group()
                            transcricao = re.search('".*"', linha).group().replace('"', '')
                            treinamento_dicionario[arquivo_audio] = transcricao



                    except UnicodeError as e:
                        pass

                    ref_arquivo.close()

        return treinamento_dicionario

    listaAudiosTreinamento = []

    def carregarAudiosTreinamento(self, nome_audio, transcricao):


        caminho_arquivos_treinamento = '../../corpus'
        for (root, dirs, arquivos) in os.walk(caminho_arquivos_treinamento):
            for arquivo in arquivos:
                if nome_audio+'.wav' in arquivo:
                    audio_time_series, sample_rate = librosa.load(os.path.join(root, nome_audio+'.wav'), sr = 16000)
                    audioObj = audio.Audio(audio_time_series, transcricao)
                    self.listaAudiosTreinamento.append(audioObj)




from joblib import Parallel, delayed
import time

if __name__ == '__main__':

    inicio = time.clock()

    treinamento = TreinamentoCNN()
    dicionario_treinamento = treinamento.obterDicionarioTreinamento()
    parallel = Parallel(backend="threading", verbose=1)
    parallel(delayed(treinamento.carregarAudiosTreinamento)(key, dicionario_treinamento[key]) for key in dicionario_treinamento)

    tempo_processamento = time.clock() - inicio

    print(treinamento.listaAudiosTreinamento[1])
    print('Tempo de processamento {}'.format(tempo_processamento))