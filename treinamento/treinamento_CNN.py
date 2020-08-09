
import os
import re
import magic

'''
REFERÊNCIAS: 

https://panda.ime.usp.br/pensepy/static/pensepy/10-Arquivos/files.html

'''
class TreinamentoCNN(object):

    def carregarAudiosTreinamento(self):

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




if __name__ == '__main__':

    treinamento = TreinamentoCNN()
    dicionario_treinamento = treinamento.carregarAudiosTreinamento()