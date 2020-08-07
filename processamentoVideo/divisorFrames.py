import multiprocessing
import os
import subprocess
from joblib import Parallel, delayed

'''
Referências:

https://wiki.python.org/moin/ParallelProcessing
https://trac.ffmpeg.org/wiki/Create%20a%20thumbnail%20image%20every%20X%20seconds%20of%20the%20video
https://github.com/khuangaf/ITRI-speech-recognition-dataset-generation/blob/master/src/split_videos.py
https://towardsdatascience.com/automatic-speech-recognition-data-collection-with-youtube-v3-api-mask-rcnn-and-google-vision-api-2370d6776109

'''
class DivisorFrames(object):

    def obterFrames(self, nomeVideo, caminhoFrames):

        caminho_video = os.path.join('../../youtubeVideos', nomeVideo)
        caminhoFramesVideo = os.path.join(caminhoFrames, nomeVideo.replace('.mp4', ''))
        comando = ['ffmpeg',  '-i', caminho_video, '-vf', 'fps=1/2', caminhoFramesVideo+'_frame_%04d_.png']
        subprocess.run(comando)


if __name__ == '__main__':

    divisor = DivisorFrames()

    path = '../../youtubeVideos'
    caminhoFrames = path + '/frames'
    os.makedirs(caminhoFrames, exist_ok=True)
    arquivosVideo = [ v for v in os.listdir(path) if v.endswith('.mp4')]

    numeroThreads = len(arquivosVideo)

    try:

        import multiprocessing as mp
        # https://www.machinelearningplus.com/python/parallel-processing-python/
        pool = mp.Pool(mp.cpu_count())
        pool.apply((divisor.obterFrames)(video, caminhoFrames) for video in arquivosVideo)
        pool.close()

        '''
        parallel = Parallel(numeroThreads, backend="threading", verbose=1)
        parallel(delayed(divisor.obterFrames)(video, caminhoFrames) for video in arquivosVideo)
        #divisor.obterFrames(arquivosVideo[0], caminhoFrames)
        '''
    except Exception as e:
        print('Não foi possível obter os frames dos vídeos: {}'.format(e))
