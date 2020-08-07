
# from textblob import TextBlob
import pytesseract
import cv2
# import multiprocessing as mp
from joblib import Parallel, delayed
import os


'''
REFERÊNCIAS: 

https://www.pyimagesearch.com/2020/08/03/tesseract-ocr-for-non-english-languages/
https://packages.debian.org/sid/graphics/tesseract-ocr-yor
https://github.com/tesseract-ocr/tesseract/tree/master/src
https://nanonets.com/blog/ocr-with-tesseract/#gettingboxesaroundtext?&utm_source=nanonets.com/blog/&utm_medium=blog&utm_content=%5BTutorial%5D%20OCR%20in%20Python%20with%20Tesseract,%20OpenCV%20and%20Pytesseract


Tesseract PSM
Page segmentation modes:
  0    Orientation and script detection (OSD) only.
  1    Automatic page segmentation with OSD.
  2    Automatic page segmentation, but no OSD, or OCR.
  3    Fully automatic page segmentation, but no OSD. (Default)
  4    Assume a single column of text of variable sizes.
  5    Assume a single uniform block of vertically aligned text.
  6    Assume a single uniform block of text.
  7    Treat the image as a single text line.
  8    Treat the image as a single word.
  9    Treat the image as a single word in a circle.
 10    Treat the image as a single character.
 11    Sparse text. Find as much text as possible in no particular order.
 12    Sparse text with OSD.
 13    Raw line. Treat the image as a single text line,
                        bypassing hacks that are Tesseract-specific.




'''
class TranscricaoVideo(object):


    def transcrever_frame(self, caminho_frame):

        image = cv2.imread(caminho_frame)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        options = "-l {} --psm {}".format('yor', '6')
        textoYoruba = pytesseract.image_to_string(rgb, config=options)
        print(textoYoruba)


    def transcrever_video(self, nomeVideo):

        caminho_frames = os.path.join('../../youtubeVideos/frames')
        frames = [os.path.join(caminho_frames, frame) for frame in os.listdir(caminho_frames) if nomeVideo in frame]

        numeroThreads = len(frames)
        parallel = Parallel(numeroThreads, backend="threading", verbose=1)
        parallel(delayed(self.transcrever_frame)(frame) for frame in frames)


if __name__ == '__main__':

    transcricao = TranscricaoVideo()

    caminho_videos = os.path.join('../../youtubeVideos/')
    videos = [video.replace('.mp4', '') for video in os.listdir(caminho_videos) if '.mp4' in video]

    '''
    # https://www.machinelearningplus.com/python/parallel-processing-python/
    pool = mp.Pool(mp.cpu_count())
    pool.apply((transcricao.transcrever_video)(video) for video in videos)
    pool.close()
    
    '''

    for video in videos:
        transcricao.transcrever_video(video)
        #break