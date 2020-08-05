
import subprocess
import requests
from bs4 import BeautifulSoup
import re
import os

class WebScraping_Bible_Yoruba(object):



    def baixarTextoYoruba(self, link):

        response = requests.get(link)
        elementos = []
        soup = BeautifulSoup(response.text, 'html.parser')
        # elementos = soup('a')
        elementos = soup.find_all('div.chapter')
        print(elementos)
        #lista_links = [href.get('href') for href in elementos]



    def baixarAudiosBibleYoruba(self, listaLinks):

        location = '/home/usuario/mestrado/yorubaSpeechRecognition/bible/audios'



        for link in listaLinks:

            #r'(http://i.imgur.com/(.*))(\?.*)?'
            pattern = re.compile(r'(https://content.cdn.dbp-prod.dbp4.org/audio/YORYOR/(YORUBSN2DA/.*mp3)(\?.*))')
            nomeArquivoAudio = pattern.match(link).group(2).replace('YORUBSN2DA/', '')
            print(nomeArquivoAudio)
            # https://stackoverflow.com/questions/2065060/using-wget-with-subprocess
            args_wget = ['wget', '-r', '-l', '1', '-p', '-P', location, link]
            wget = subprocess.run(args_wget,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                universal_newlines=True )

            print(wget.stdout)

            #break


    def carregarListaLinks(self):

        listaLinks = []
        arquivos = os.listdir('../bible')

        for arquivoLink in arquivos:
                if '_linkAudio_' in arquivoLink:
                    arquivo = open('../bible/'+arquivoLink, 'r', encoding="utf8")
                    listaLinks.append(arquivo.read())

        return listaLinks


if __name__ == '__main__':

    from scraping import links_odu, linksBible

    scraper = WebScraping_Bible_Yoruba()
    #listaLinks = linksBible.EnumListaLinksBible().listaLinks

    listaLinks = scraper.carregarListaLinks();

    print(listaLinks)
    try:
        scraper.baixarAudiosBibleYoruba(listaLinks)
    except:
        pass

    #link = 'https://live.bible.is/bible/YORYOR/MAT/1'
    #scraper.baixarTextoYoruba(link);