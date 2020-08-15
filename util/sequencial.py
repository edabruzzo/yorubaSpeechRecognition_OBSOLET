
class Sequencial(object):



    def executarMetodoEmSequencia(self, funcao, lista):
        for elemento in lista:
            funcao(elemento)

