
from multiprocessing import Process
from monitoramento import monitoramento_PROPRIETARY as monitoramento_memoria
# from monitoramento import monitoramento_memoria
import datetime
import os


class Monitoramento(object):


    def monitorar_memoria(self, funcao, configuracao_paralelizacao, argumentos=None):

        '''

        # https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html


        Importei o arquivo para monitoramento de memória do sequinte projeto:
        https://github.com/astrofrog/psrecord/blob/master/psrecord/main.py

        Arquivo importado no diretório: /usr/lib/python3.6/
        Nome: monitoramento_memoria.py

        '''
        if argumentos is not None:
            p1 = Process(target=funcao, args=argumentos)
        else:
            p1 = Process(target=funcao)
        p1.start()
        pid_subProcess = p1.pid

        if pid_subProcess is not None:
            pid = pid_subProcess
        else:
            pid = os.getpid()

        path = '/home/usuario/mestrado/yorubaSpeechRecognition_RECOVERY/logs/monitoramento_memoria/'

        arquivoLog = os.path.join(path, f'yorubaSpeechRecognition__'
                                        f'{str(datetime.datetime.today())}__'
                                        f'{str(datetime.time())}__')
        '''
         Processo rodando em paralelo para monitora uso de memória, CPU, tempo decorrido no pré-processamento    
        '''
        p2 = Process(target=monitoramento_memoria.monitor,
                     args=(configuracao_paralelizacao,
                           pid,
                           arquivoLog + '.txt',
                           arquivoLog + '.png'))
        p2.start()
