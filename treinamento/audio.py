class Audio(object):

    def __init__(self, time_series, transcricao):

        self.__time_series = time_series
        self.__transcricao = transcricao

    @property
    def time_series(self):
        return self.__time_series

    @time_series.setter
    def time_series(self, time_series):
        self.__time_series = time_series


    @property
    def transcricao(self):
        return self.__transcricao

    @time_series.setter
    def transcricao(self, transcricao):
        self.__transcricao = transcricao
