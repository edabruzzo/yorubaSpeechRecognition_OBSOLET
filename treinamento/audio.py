'''
REFERÊNCIA:

https://medium.com/python-weekly-brazil/chega-de-get-e-set-atributos-din%C3%A2micos-em-python-2e79cf7ec196


'''

class Audio(object):

    def __init__(self, nome_arquivo, caminho_arquivo, transcricao, log_energy, label_encoded):

        self.__log_energy = log_energy
        self.__label_encoded = label_encoded
        self.__nome_arquivo = nome_arquivo
        self.__caminho_arquivo = caminho_arquivo
        self.__transcricao = transcricao


    @property
    def caminho_arquivo(self):
        return self.__caminho_arquivo

    @caminho_arquivo.setter
    def caminho_arquivo(self, caminho_arquivo):
        self.__caminho_arquivo = caminho_arquivo



    @property
    def transcricao(self):
        return self.__transcricao

    @transcricao.setter
    def transcricao(self, transcricao):
        self.__transcricao = transcricao

    @property
    def nome_arquivo(self):
        return self.__nome_arquivo

    @nome_arquivo.setter
    def nome_arquivo(self, nome_arquivo):
        self.__nome_arquivo = nome_arquivo


    @property
    def log_energy(self):
        return self.__log_energy

    @log_energy.setter
    def log_energy(self, log_energy):
        self.__log_energy = log_energy


    @property
    def label_encoded(self):
        return self.__label_encoded

    @label_encoded.setter
    def label_encoded(self, label_encoded):
        self.__label_encoded = label_encoded
