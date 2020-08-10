'''
REFERÊNCIA:

https://medium.com/python-weekly-brazil/chega-de-get-e-set-atributos-din%C3%A2micos-em-python-2e79cf7ec196


'''

class Audio(object):

    def __init__(self, log_energy, label_encoded):

        self.__log_energy = log_energy
        self.__label_encoded = label_encoded

    @property
    def log_energy(self):
        return self.__log_energy

    @log_energy.setter
    def mfcc(self, log_energy):
        self.__log_energy = log_energy


    @property
    def label_encoded(self):
        return self.__label_encoded

    @label_encoded.setter
    def transcricao(self, label_encoded):
        self.__label_encoded = label_encoded
