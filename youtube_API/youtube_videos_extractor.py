# -*- coding: utf-8 -*-

'''
 https://developers.google.com/api-client-library/dotnet/guide/aaa_client_secrets
 https://console.developers.google.com/apis/credentials?project=uplifted-env-285612
 https://developers.google.com/explorer-help/guides/code_samples#python
 https://developers.google.com/youtube/v3/quickstart/python
 https://developers.google.com/youtube/v3/code_samples/code_snippets?apix=true
 https://github.com/khuangaf/ITRI-speech-recognition-dataset-generation/blob/master/src/download_videos.py
 https://towardsdatascience.com/automatic-speech-recognition-data-collection-with-youtube-v3-api-mask-rcnn-and-google-vision-api-2370d6776109
 https://github.com/nikhilkumarsingh/YouTubeAPI-Examples/blob/master/4.Channel-Vids.ipynb
 https://pypi.org/project/pytube3/
 https://github.com/erykml/medium_articles/blob/master/Computer%20Vision/downloading_youtube_videos.ipynb
 https://towardsdatascience.com/the-easiest-way-to-download-youtube-videos-using-python-2640958318ab
 https://pypi.org/project/pytube3/
 https://stackoverflow.com/questions/47420304/download-video-in-mp3-format-using-pytube

'''

from googleapiclient.discovery import build
from pytube import YouTube
import subprocess
import os


class YoutubeVideosExtractor(object):

    # developer keys for Youtube V3 API
    DEVELOPER_KEY_FILE = "../../yorubaAPIKey"
    DEVELOPER_KEY = open(DEVELOPER_KEY_FILE, 'r', encoding="utf8").read()
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    client_secrets_file = "../../client_secret.json"

    # creating youtube resource object for interacting with api
    youtube = build(YOUTUBE_API_SERVICE_NAME,
                    YOUTUBE_API_VERSION,
                    developerKey=DEVELOPER_KEY)

    urlBaseYoutube = "https://www.youtube.com/watch?v="



    def obterVideosYoruba(self, nomeLista=None, id_lista=None ):

        # Disable OAuthlib's HTTPS verification when running locally.
        # *DO NOT* leave this option enabled in production.
        #os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

        # search for the first playlist result given a drama name
        search_response = self.youtube.search().list(q=nomeLista, type="playlist", part="id", maxResults=1).execute()
        result = search_response.get("items", [])
        playlist_id = result[0]['id']['playlistId']
        # search for all the videos given a playlist id
        search_response = self.youtube.playlistItems().list(part='contentDetails', maxResults=50,
                                                       playlistId=playlist_id).execute()
        videos = search_response['items']

        for video in videos:
            video_id = video['contentDetails']['videoId']
            self.listaVideos.append(self.urlBaseYoutube + video_id)


    def obterVideosCanal(self, id_canal):

        # get Uploads playlist id
        resposta = self.youtube.channels().list(id=id_canal,
                                                part='contentDetails').execute()

        playlist_id = resposta['items'][0]['contentDetails']['relatedPlaylists']['uploads']

        next_page_token = None

        lista_videos = []

        while 1:
            response = self.youtube.playlistItems().list(playlistId=playlist_id,
                                               part='snippet',
                                               maxResults=1000,
                                               pageToken=next_page_token).execute()
            lista_videos += response['items']
            nextPageToken = response.get('nextPageToken')

            if nextPageToken is None:
                break

        return lista_videos


    def extrairAudioVideo(self, fileName):

        path = '../../youtubeVideos'
        video_path = os.path.join(path, fileName)
        audio_path = video_path.replace('mp4', 'mp3')

        comando = ['ffmpeg', '-i', video_path, '-f', 'mp3' ,'-ab', '192000', '-vn', '-y', audio_path]

        # https://janakiev.com/blog/python-shell-commands/
        ffmpeg = subprocess.run(comando,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                universal_newlines=True
                                )

        for line in ffmpeg.stdout:
            print(line.strip())



    def downloadVideoYoutube(self, idVideo):

        try:

            path = '../../youtubeVideos/'

            try:

                yt = YouTube(self.urlBaseYoutube + idVideo)
                # https://pypi.org/project/pytube3/
                streams = yt.streams
                video = streams.filter(progressive=True, file_extension='mp4').order_by('resolution')[-1]
                video.download(path)
                self.extrairAudioVideo(video.default_filename)
                print("Vídeo baixado: {id} ".format(id=idVideo))

                #return True

            except Exception as e:
                print("- {exception}".format(exception=e))
                print("- {id} Vídeo não pôde ser baixado".format(id=idVideo))
                #return False
        except Exception as e:
            print('Não foi possível efetuar o download do vídeo: {}'.format(e))
            #return False




if __name__ == '__main__':

    youtubeApi = YoutubeVideosExtractor()
    #nomeLista = ''
    #youtubeApi.obterVideosYoruba(nomeLista)

    #youtubeApi.listaVideos.append(youtubeApi.urlBaseYoutube + '38uZCB-KCx0')
    #youtubeApi.listaVideos.append(youtubeApi.urlBaseYoutube + 'ivtHboJa16w')
    #print(youtubeApi.listaVideos)

    #https://www.youtube.com/channel/UCzpkJhafrktQJDDqN5Rxg-Q/playlists
    #Nalingo Naija
    listaVideos = youtubeApi.obterVideosCanal('UCzpkJhafrktQJDDqN5Rxg-Q')
    print('Foram obtidos {} vídeos'.format(len(listaVideos)))


    for video in listaVideos:
        videoId = video['snippet']['resourceId']['videoId']
        print(video['snippet']['title'])
        youtubeApi.downloadVideoYoutube(videoId)

    # https://www.youtube.com/watch?v=5xxEvVxXyMY - Itan Jesu fiimu • Yoruba Yooba Yariba • Story of Jesus Movie
    # Necessário verificar Evangelho de João
    # This film features life and toils of Jesus Christ, depicted as written in the Gospel of John.
    # youtubeApi.downloadVideoYoutube('5xxEvVxXyMY')

    # https://www.youtube.com/watch?v=aTXbOzzcatw
    # "New World Order Bible Versions" (Yoruba subtitles)

    #https://www.youtube.com/watch?v=eYaDMptJQYE&list=PLC0Qb6b17wHHtOziRpgOkbDnqtZANSyxe&index=9