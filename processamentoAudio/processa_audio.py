import webrtcvad

'''
https://github.com/wiseman/py-webrtcvad/blob/master/example.py
https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/44091
https://www.kaggle.com/davids1992/speech-representation-and-data-exploration
https://colab.research.google.com/drive/1V2wpsagje7pOQzubv3OQWNFsev4SM7tR#scrollTo=LxhtnYGGxBvc
https://unix.stackexchange.com/questions/280767/how-do-i-split-an-audio-file-into-multiple


'''



from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

class ProcessamentoAudio(object):

    def obter_speech_from_wav_file_pydub(self, directory, file):

        sound_file = AudioSegment.from_wav(os.path.join(directory, file))
        audio_chunks = split_on_silence(sound_file,
                                        # duração do silêncio em milisegundos
                                        min_silence_len=100,

                                        # consider it silent if quieter than -16 dBFS
                                        silence_thresh=-16 )

        path = os.path.join(directory, "chunks")
        os.makedirs(path, exist_ok=True)

        for i, chunk in enumerate(audio_chunks):
            out_file = f'chunk{i}.wav'
            print('Gravando chunk %s' % out_file)
            chunk.export(os.path.join(path, out_file), format="wav")


if __name__ == '__main__':

    path = '/home/usuario/mestrado/bible/audios/content.cdn.dbp-prod.dbp4.org/audio/YORYOR/YORUBSN2DA/teste'
    file = 'teste.wav'
    ProcessamentoAudio().obter_speech_from_wav_file_pydub(path, file)
