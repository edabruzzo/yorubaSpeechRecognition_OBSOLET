
import subprocess

class WebScraping_Bible_Yoruba(object):

    def baixarAudiosBibleYoruba(self, listaLinks):

        location = '/home/usuario/mestrado/yorubaSpeechRecognition/bible'



        for link in listaLinks:
            # https://stackoverflow.com/questions/2065060/using-wget-with-subprocess
            args = ['wget', '-r', '-l', '1', '-p', '-P', location, link]
            #args = ['pwd']
            wget = subprocess.run(args,
                                  stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  universal_newlines=True
                                  )

            print(wget.stdout)

            break


if __name__ == '__main__':

    from scraping import links_odu, linksBible

    scraper = WebScraping_Bible_Yoruba()
    # scraper.obterLinksVideosHarvard()
    listaLinks = linksBible.EnumListaLinksBible().listaLinks
    scraper.baixarAudiosBibleYoruba(listaLinks)