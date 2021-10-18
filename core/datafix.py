import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import chardet
from io import StringIO
# from django.core.exceptions import ValidationError
import json
from io import BytesIO


def fileToJson(data) :
    """
    xlsx, csv 파일을 df -> json으로 변환
    """
    dataname = data.filename
    # print(type(data))
    # print(dataname)
    # print(data)


    if dataname.endswith('.csv'):
        df = pd.read_csv(data, encoding='cp949')
        
        # print(rawdata)
        # result = chardet.detect(rawdata)
        # print(result)
        # encoder = result['encoding']
        # file = StringIO(rawdata.decode(encoder))
        # df = pd.read_csv(file)
        # try :
        #     rawdata = data.read()
        #     result = chardet.detect(rawdata)
        #     print(result)
        #     encoder = result['encoding'].lower()
        #     file = StringIO(rawdata.decode(encoder))
        #     df = pd.read_csv(file)
        # except :
        #     pass
    else:
        df = pd.read_excel(data, encoding='cp949', engine='openpyxl')#

    df_js = df.to_json(orient='split')
    print(f'df_js type {type(df_js)}')
    desc_js = getDescribe(df_js)
    corr_js = getCorr(df_js)

    cols = len(df.columns.tolist())

    return df_js, desc_js, corr_js, cols


def missingFix(data, ifnan, col) :
    """
    결측치 처리법(예시)
    dropna0 : 셀 Drop
    dropna1 : 컬럼 Drop
    fillmedi : 중앙값으로 채우기
    fillmean : 평균값으로 채우기
    fillpad : 이전값으로 채우기
    fillbfill : 이후값으로 채우기
    else : 특정값으로 채우기
    """
    df = pd.read_json(data, orient='split')
    # 결측치 처리
    if ifnan == 'dropna0':
        df[col].dropna(inplace=True)
    elif ifnan == 'dropna1':
        if any(df[col].isnull()) :
            df.drop([col], axis=1, inplace=True)
    elif ifnan == 'fillmedi':
        try :
            df[col].fillna(df[col].median(), inplace=True)
        except :
            raise Exception('숫자형 컬럼에만 적용할 수 있습니다')
    elif ifnan == 'fillmean':
        try :
            df[col].fillna(df[col].mean(), inplace=True)
        except :
            raise Exception('숫자형 컬럼에만 적용할 수 있습니다')
    elif ifnan == 'fillpad':
        df[col].fillna(method='pad', inplace=True)
    elif ifnan == 'fillbfill':
        df[col].fillna(method='bfill', inplace=True)
    else:
        try :
            # Data type에 따라 변환하는 방법 모색(dtypes 활용은 실패함)
            df[col].fillna(float(ifnan), inplace=True)
        except :
            df[col].fillna(ifnan, inplace=True)
    df_js = df.to_json(orient='split')

    return df_js


def getDescribe(data) :
    """
    완성 데이터셋의 Describe를 위한 describe DF 반환
    """
    # describe 항목 순서 : ['COUNT', 'MIN', 'MAX', 'MEAN', 'STD', '50%', '25%', '75%']

    df = pd.read_json(data, orient='split')
    desc = pd.DataFrame(df.columns.tolist(), columns=['column'], index=df.columns.tolist()).T

    means = []
    stds = []
    q1 = []
    q2 = []
    q3 = []
    for i in df.columns:
        try:
            means.append(df[i].mean())
            stds.append(df[i].std())
            q1.append(df[i].quantile(.25))
            q2.append(df[i].quantile(.50))
            q3.append(df[i].quantile(.75))
        except:
            means.append('NaN')
            stds.append('NaN')
            q1.append('NaN')
            q2.append('NaN')
            q3.append('NaN')

    desc.loc['type', :] = df.dtypes.values.astype('str')
    desc.loc['count', :] = df.count().values.astype('int64')
    desc.loc['missing', :] = df[df.isnull() == True].count().values.astype('str')
    desc.loc['mean', :] = means
    desc.loc['std', :] = stds
    desc.loc['min', :] = df.min().values.astype('str')
    desc.loc['25%', :] = q1
    desc.loc['50%', :] = q2
    desc.loc['75%', :] = q3
    desc.loc['max', :] = df.max().values.astype('str')

    desc_js = desc.to_json(orient='split')
    desc_js = json.loads(desc_js)

    return desc_js


def getCorr(data) :
    """
    완성 데이터셋의 히트맵을 위핸 corr DF 반환
    """
    df = pd.read_json(data, orient='split')
    pre_corr = df.corr()
    cols = pre_corr.columns.tolist()
    corr = pd.DataFrame(df.corr(), columns=cols)
    corr_js = corr.to_json(orient='split')
    corr_js = json.loads(corr_js)

    return corr_js


def getColumns(data) :
    df = pd.read_json(data, orient='split')
    cols = df.columns.tolist()
    return cols


def getNans(data) :
    df = pd.read_json(data, orient='split')
    nans = df.isnull()
    nans_js = nans.to_json(orient='split')

    nans_js = json.loads(nans_js)

    return nans_js


def getHead(data) :
    df = pd.read_json(data, orient='split')
    head = df.head(10)
    head_js = head.to_json(orient='split')

    head_js = json.loads(head_js)

    return head_js


def searchOut(data, col, val) :
    df = pd.read_json(data, orient='split')
    try :
        val = float(val)
    except :
        val = val

    df = pd.DataFrame(df[df[col]==val][col], columns=[col])
    df_js = df.to_json(orient='split')

    df_js = json.loads(df_js)

    return df_js


def setOut(data, fixdata) :
    df = pd.read_json(data, orient='split')
    try :
        df_fix = pd.DataFrame(fixdata['data'][0], columns=fixdata['columns'], index=fixdata['index'])
    except :
        df_fix = pd.DataFrame(fixdata['data'], columns=fixdata['columns'], index=fixdata['index'])
    df_col = df_fix.columns.tolist()[0]
    for i in df_fix.index :
        df.loc[i,[df_col]] = df_fix.loc[i,[df_col]].item()
    df_fix = df.loc[df_fix.index, [df_col]]
    df_js = df.to_json(orient='split')
    df_fix_js = df_fix.to_json(orient='split')

    df_fix_js = json.loads(df_fix_js)

    return df_js, df_fix_js


def getEncoder(data, encoder, col):
    """
    데이터와 컬럼명을 전달받아 Ordinal / OneHot 두 가지 중 선택하여 인코더 반영
    """
    print('data: ', type(data))
    # .fillna() 임의로 추가 10/15
    data = pd.read_json(data, orient='split').fillna(0)
    print(data.columns)
    if encoder == 'OrdinalEncoder' :
        cols = data.columns.tolist()
        ord_enc = pd.DataFrame(OrdinalEncoder().fit_transform(data[[col]]), columns=[col])
        data.drop([col], axis=1, inplace=True)
        data = pd.concat([data, ord_enc], axis=1)
        data = data[cols]

    else :
        v_list = data[col].unique().tolist()
        print(v_list)
        print(type(col))
        hot_enc = pd.DataFrame(OneHotEncoder().fit_transform(data[[col]].values.reshape(-1, 1)).toarray())
        # 10/15 v -> str(v)
        hot_enc.columns = [col + '_' + str(v) for v in v_list]
        data.drop([col], axis=1, inplace=True)
        data = pd.concat([data, hot_enc], axis=1)

    df = data.to_json(orient='split')

    return df


def getFilesize(data) :
    """
    파일사이즈 구하기
    """
    df = pd.read_json(data, orient='split')
    size = df.memory_usage(index=True, deep=True).sum()
    unit = 'Bytes'
    if size > 1024:
        size = round(size / 1024, 2)
        unit = 'KB'
        if size > 1024:
            size = round(size / 1024, 2)
            unit = 'MB'
    size = str(size) + ' ' + unit

    return size
