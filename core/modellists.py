from .Util import *
import os
from pathlib import Path



@singleton
class GetModel :

    ifnanlist = []
    scalerlist = []
    encoderlist = []
    modellist = {}
    finPath = ''

    def __init__(self, *args, **kwargs):
        self.ifnanlist = {'label' : ['특정값으로 변경', '이전값으로 변경', '이후값으로 변경', '중간값으로 변경', '평균값으로 변경', '셀 삭제', '행 삭제'],
                          'value' : ['else', 'fillpad', 'fillbfill', 'fillmedi', 'fillmean', 'dropna0', 'dropna1']}
        self.scalerlist = ['적용 안 함', 'StandardScaler', 'RobustScaler', 'MinMaxScaler', 'Normalizer']
        self.encoderlist = ['OrdinalEncoder', 'OneHotEncoder']
        self.modellist = {'modelname' : ['선형 회귀', '랜덤포레스트 회귀', 'XGBoost 회귀', 'MLP 회귀', '로지스틱 회귀', '랜덤포레스트 분류', 'XGBoost 분류', 'MLP 분류'],
                          'about' : ['선형 회귀 설명',
                                     '랜덤포레스트 회귀 설명',
                                     'XGBoost 회귀 설명',
                                     'MLP 회귀 설명',
                                     '로지스틱 회귀 설명',
                                     '랜덤포레스트 분류 설명',
                                     'XGBoost 분류 설명',
                                     'MLP 분류 설명'],
                          'params' : [[],
                                      ['트리의 갯수', '트리의 최대 깊이'],
                                      ['트리의 갯수', '트리의 최대 깊이'],
                                      [],
                                      [],
                                      ['트리의 갯수', '트리의 최대 깊이'],
                                      ['트리의 갯수', '트리의 최대 깊이'],
                                      []]}


        dir = Path(__file__).resolve().parent
        getPath = os.path.join(dir, 'graphviz-2.38')
        self.finPath = os.path.abspath(os.path.join(getPath, 'release', 'bin'))
        if self.finPath not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + self.finPath