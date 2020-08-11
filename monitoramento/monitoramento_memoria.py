#https://sobrelinux.info/questions/769806/how-to-monitor-cpu-memory-usage-of-a-single-process
#
'''
import psutil

print(psutil.cpu_percent())                    # Em porcentagem, uso da CPU
print(psutil.virtual_memory()._asdict())       # Em dicionário informações de memória

'''