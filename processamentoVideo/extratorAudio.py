import os
import subprocess


class ExtratorAudio(object):


    def extrairAudioVideos(self, caminho):
        '''
        https://towardsdatascience.com/automatic-speech-recognition-data-collection-with-youtube-v3-api-mask-rcnn-and-google-vision-api-2370d6776109

        '''

        for root, diretorios, files in os.walk(caminho):
            for file in files:

                if file.endswith(".mp4"):

                    if os.path.exists(root + '/sem_legenda_teste/' + file) :
                        video_path = os.path.join(root + '/sem_legenda_teste/', file)
                        audio_path = os.path.join(root + '/sem_legenda_teste/', file).replace('.mp4', '.mp3')
                    else:
                        video_path = os.path.join(root, file)
                        audio_path = os.path.join(root, file).replace('.mp4', '.mp3')


                    if not os.path.exists(audio_path) :


                        print('Extraindo áudio do vídeo %s para o arquivo %s' % (video_path, audio_path))

                        # -i: caminho para os arquivos de vídeo.
                        # -f:  mp3 diz ao ffmpeg que a saída será em formato mp3.
                        # -ab 192000: a saída será em 192Kbps
                        # -vn : para dizer que não queremos vídeo.
                        comando = f'ffmpeg -i {video_path} -f mp3 -ab 192000 -vn -y {audio_path}'.split(" ")

                        #https://janakiev.com/blog/python-shell-commands/
                        ffmpeg = subprocess.run(comando,
                                       stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       universal_newlines=True
                                       )

                        for line in ffmpeg.stdout:
                            print(line.strip())
                    else:
                        print('File %s already exists' % audio_path)




if __name__ == '__main__':
    extrator = ExtratorAudio()
    extrator.extrairAudioVideos('../../videos/sem_legenda_teste/')
    extrator.extrairAudioVideos('../../videos/')
