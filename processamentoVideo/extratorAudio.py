import os
import subprocess


class ExtratorAudio(object):


    def extrairAudioVideos(self):
        '''
        https://towardsdatascience.com/automatic-speech-recognition-data-collection-with-youtube-v3-api-mask-rcnn-and-google-vision-api-2370d6776109

        '''

        for file in os.listdir("../../videos"):
            if file.endswith(".mp4"):
                print(os.path.join("../../videos", file))
                video_path = os.path.join("../../videos", file)
                audio_path = os.path.join("../../videos", file).replace('.mp4', '.mp3')
                # -i: it is the path to the input file. The second option -f mp3 tells ffmpeg that the ouput is in mp3 format.
                # -ab 192000: we want the output to be encoded at 192Kbps
                # -vn :we dont want video.
                # execute the command
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





if __name__ == '__main__':
    extrator = ExtratorAudio()
    extrator.extrairAudioVideos()