from joblib import Parallel, delayed

class Paralelizacao(object):

    def __init__(self, n_jobs=4, verbose=5, backend='multiprocessing'):
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.backend=backend


    def executarMetodoParalelo(self, funcao, lista):

        parallel = Parallel(n_jobs=self.n_jobs, backend=self.backend, verbose=self.verbose)
        parallel(delayed(funcao)(elemento) for elemento in lista)
