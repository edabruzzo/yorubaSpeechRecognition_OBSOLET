# -*- coding: utf-8 -*-

# Sample Python code for youtube.channels.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/guides/code_samples#python
#https://developers.google.com/youtube/v3/quickstart/python

import os
from googleapiclient.discovery import build



class YoutubeVideosExtractor(object):
    # developer keys for Youtube V3 API
    DEVELOPER_KEY_FILE = "../../yorubaAPIKey"

    DEVELOPER_KEY = open(DEVELOPER_KEY_FILE, 'r', encoding="utf8").read()

    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"

    # creating youtube resource object for interacting with api
    youtube = build(YOUTUBE_API_SERVICE_NAME,
                    YOUTUBE_API_VERSION,
                    developerKey=DEVELOPER_KEY)

    urlBaseYoutube = "https://www.youtube.com/watch?v="
    listaVideos = []


    def obterVideosYoruba(self, nomeLista=None, id_lista=None ):

        # Disable OAuthlib's HTTPS verification when running locally.
        # *DO NOT* leave this option enabled in production.
        #os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

        api_service_name = "youtube"
        api_version = "v3"
        client_secrets_file = "../../client_secret.json"
        API_KEY = "../../yorubaAPIKey"
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
                                               maxResults=100,
                                               pageToken=next_page_token).execute()
            lista_videos += response['items']
            nextPageToken = response.get('nextPageToken')

            if nextPageToken is None:
                break

        return lista_videos


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

    for video in listaVideos:
        videoId = video['snippet']['resourceId']['videoId']
        youtubeApi.listaVideos.append(youtubeApi.urlBaseYoutube + videoId)