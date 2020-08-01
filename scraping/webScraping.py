
import random
import requests
from bs4 import BeautifulSoup
import subprocess


class Web_Scraping(object):

    listaGloballinks = []
    urlAskDL = 'http://ask-dl.fas.harvard.edu/content/'

    def obterLinksVideosHarvard(self):

        url_base='http://ask-dl.fas.harvard.edu/odu-ifa/content/20-oy-ku-meji'

        response = requests.get(url_base, headers=self.obterHeaders())
        elementos = []
        soup = BeautifulSoup(response.text, 'html.parser')
        #elementos = soup('a')
        elementos = soup.find_all('ul')
        lista_links = [href.get('href') for href in elementos]



        for link in lista_links:

            linkAtual = url_base + link
            response_Odu = requests.get('http://ask-dl.fas.harvard.edu/odu-ifa/content/20-oy-ku-meji', headers=self.obterHeaders())

            if '/content/' in link and link != linkAtual.replace(url_base, ''):

                self.listaGloballinks.append(link)

                if link not in self.listaGloballinks:

                    html = BeautifulSoup(response_Odu.text, 'html.parser')
                    #conteudoYorubaDiv = [div for div in soup.find_all(class_="left odu-transcription rounded-small")]
                    #https://www.crummy.com/software/BeautifulSoup/bs4/doc/#find
                    #conteudoYorubaDiv = [div for div in soup.select('div.left.odu-transcription.rounded-small')]
                    #paragrafos = [paragr.find_all('p') for paragr in conteudoYorubaDiv ]
                    #print(paragrafos)

                    elementos_Odu = html('a')
                    listaParciallinks = [href_Odu.get('href') for href_Odu in elementos_Odu]
                    print(listaParciallinks)
                    self.listaGloballinks.append(listaParciallinks)

    def obterHeaders(self):

        listaUserAgents = [
            # Chrome
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
            'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
            'Mozilla/5.0 (Windows NT 5.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
            'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
            'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
            'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
            'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',

            # Firefox
            'Mozilla/4.0 (compatible; MSIE 9.0; Windows NT 6.1)',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36',
            'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
            'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)',
            'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
            'Mozilla/5.0 (Windows NT 6.2; WOW64; Trident/7.0; rv:11.0) like Gecko',
            'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko',
            'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.0; Trident/5.0)',
            'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko',
            'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)',
            'Mozilla/5.0 (Windows NT 6.1; Win64; x64; Trident/7.0; rv:11.0) like Gecko',
            'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)',
            'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)',
            'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 5.1; Trident/4.0; .NET CLR 2.0.50727; .NET CLR 3.0.4506.2152; .NET CLR 3.5.30729)']

        user_agent = random.choice(listaUserAgents)
        headers = {'User-Agent': user_agent,
                       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                   'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                   'Accept-Encoding': 'none',
                   # 'Accept-Language': 'en-US,en;q=0.8',
                   'Connection': 'keep-alive'}

        print("User-Agent enviado no request:%s\n" % (user_agent))
        print("-------------------\n\n")
        return headers



    def baixarTranscricoes_CURL(self, listaLinks):

        for link in listaLinks:

            comando = f'curl  -o ./{link}.html {self.urlAskDL + link}'.split(" ")
            curl = subprocess.run(comando,
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True
                                    )


            print(curl.stdout)

            break


if __name__ == '__main__':

    from scraping import links_odu

    scraper = Web_Scraping()
    #scraper.obterLinksVideosHarvard()
    listaLinks = links_odu.EnumListaLinks().listaLinks
    scraper.baixarTranscricoes_CURL(listaLinks)
