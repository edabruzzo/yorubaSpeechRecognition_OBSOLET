
class Sequencial(object):



    def executarMetodoEmSequencia(self, funcao, lista):
        print('Iniciando execução sequencial da função: ' + str(funcao))
        for elemento in lista:
            funcao(elemento)

