
'''

https://pypi.org/project/psrecord/
https://sobrelinux.info/questions/769806/how-to-monitor-cpu-memory-usage-of-a-single-process

Problema na instalação do psutil com pip
https://sempreupdate.com.br/como-corrigir-fatal-error-python-h-no-such-file-or-directory/
*** Solução: sudo apt-get install python3-dev

https://unix.stackexchange.com/questions/554/how-to-monitor-cpu-memory-usage-of-a-single-process/414770

https://unix.stackexchange.com/questions/554/how-to-monitor-cpu-memory-usage-of-a-single-process/414770

pip install psrecord                             # local user install
sudo apt-get install python-matplotlib python-tk # for plotting; or via pip
For single process it's the following (stopped by Ctrl+C):

psrecord $(pgrep proc-name1) --interval 1 --plot ./monitoramento/plot1.png --include-children
For several processes the following script is helpful to synchronise the charts:

#!/bin/bash
psrecord $(pgrep proc-name1) --interval 1 --duration 60 --plot plot1.png &
P1=$!
psrecord $(pgrep proc-name2) --interval 1 --duration 60 --plot plot2.png &
P2=$!
wait $P1 $P2
echo 'Done'


'''




import os
import psutil
import subprocess
import datetime


import time


def monitorar(pid_python, logfile, plot):

    monitoramento_path = '/monitoramento/monitoramento_PROPRIETARY.py'

    '''
    #duration=None,
    #interval=None,
    #include_children=False
    
    '''

    comando = f'python ' \
              f'{monitoramento_path} ' \
              f'{pid_python} ' \
              f'--plot {plot} ' \
              f'--logfile {logfile} '

    subprocess.run(comando)