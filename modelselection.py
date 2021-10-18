from flask import Blueprint, render_template, jsonify, request
from models import Data, Temp2
from extension import db
from core import modellists
from core.datafix import *
from core.modeltrain import *

modelselection = Blueprint("modelselection", __name__)

@modelselection.route('choosecategory/<data_id>/')
def choosecategory(data_id):
    """학습유형 선택"""
    data = Data.query.filter_by(idx=data_id).first()
    if data == None:
        return jsonify({"message": "dataset not found"})

    response = {
        'pid': data.project.idx,
        'did': data.idx,
        'pname': data.project.pname,
        'dname': data.dname,
        'filename': data.filename
    }

    return jsonify(response)

@modelselection.route('deletetemp/<tid>/', methods=["delete"])
def deletetemp(tid):
    """해당 페이지를 벗어날 때 Temp2에 저장된 내용을 삭제
    시리얼라이저는 Temp것 그대로 사용"""
    data = Temp2.object.filter_by(idx=tid).first()

    if data == None:
        return jsonify({"message": "dataset not found"})

    did = data.dataset.idx
    pid = data.dataset.project.idx

    #for all records
    db.session.query(data).delete()
    db.session.commit()

    response = {
        'pid': pid,
        'did': did,
        'message': "Success"
    }

    return jsonify(response)
    
@modelselection.route('/presplit', methods=['POST'])
def presplit(): # 10/15 테스트 완료
    """데이터셋 분리/스케일러 적용 페이지 진입(이미 진행하던 내용이 있을 경우 Temp2에 있는 정보 리턴)"""
    
    mdl = modellists.GetModel()
    scalerlist = mdl.scalerlist
    data = request.get_json(force=True)['data']
    did = data['did']
    dobj = Data.query.filter_by(idx=did).first()
    cols = getColumns(dobj.data)
    
    if 'tid' in data:
        tid = data['tid']
        file_instance = Temp2.query.filter_by(idx=tid).first()
    else:
        file_instance = Temp2 (
            dataset = dobj.idx,
            xcol = [],
            ycol = "",
            trainsize = 8,
            trainsize2 = 8,
            testsize = 2,
            validsize = 2,
            scaler = 'no_scale'
        )

        db.session.add(file_instance)
        db.session.commit()

    response = {
        'tid' : file_instance.idx,
        'pid' : dobj.project_id,
        'did': dobj.idx,
        'pname' : dobj.project.pname,
        'dname' : dobj.dname,
        'filename' : dobj.filename,
        'columns' : cols,
        'scalerlist' : scalerlist,
        'xcol' : file_instance.xcol,
        'ycol' : file_instance.ycol,
        'trainsize' : file_instance.trainsize,
        'trainsize2' : file_instance.trainsize2,
        'testsize' : file_instance.testsize,
        'validsize' : file_instance.validsize,
        'scaler' : '적용 안 함'
        }

    return response

@modelselection.route('/savemodel', methods=['POST'])
def savemodel():
    """모델 저장"""
    data = request.get_json(force=True)
    tid = data['tid']
    did = Temp2.query.filter(idx=tid)
    instance = Data.query.filter(idx=did)
    df = instance.data
    prj = instance.project
    filename = instance.filename.split('.')[0]
    scaler = data['scaler']
    xcol = data['xcol']
    ycol = data['ycol']
    test = data['testsize'] / 10
    valid = data['validsize'] / 10

    X, Y = getXandY(df, xcol, ycol)
    X_scaled = getScaler(X, scaler)
    X_trm, Y_tr, X_te, Y_te, X_va, Y_va = splitData(X_scaled, Y, test, valid)

    trainamount = getAmount(X_tr)
    testamount = getAmount(X_te)
    validamount = getAmount(X_va)
    mdl = modellists.GetModel()

    data['did'] = did
    data['projects'] = prj
    data['columns'] = getColumns(df)
    data['scalerlist'] = mdl.scalerlist
    data['xmemo'] = 'Test_X'
    data['ymemo'] = 'Test_Y'
    data['xfilename'] = filename + '_test_x.csv'
    data['yfilename'] = filename + '_test_y.csv'
    data['xfilesize'] = getFilesize(X_te)
    data['yfilesize'] = getFilesize(Y_te)
    data['xcolumncount'] = len(getColumns(X_te))
    data['ycolumncount'] = len(getColumns(Y_te))
    data['xtest'] = X_te
    data['ytest'] = Y_te
    data['trainamount'] = trainamount
    data['testamount'] = testamount
    data['validamount'] = validamount
    
@modelselection.route('/savesplit', methods=['POST'])
def savesplit():
    """테스트셋 이름 2개 전달받아 Data에 저장"""
    data = request.get_json(force=True)
    tid = data.get('tid')
    did = Temp2.objects.filter(idx=tid).get().dataset.idx
    obj = Data.objects.filter(idx=did).get()
    df = obj.data
    prj = obj.project
    filename = obj.filename.split('.')[0]
    scaler = data.get('scaler')
    xcol = data.get('xcol')
    ycol = data.get('ycol')
    test = data.get('testsize') / 10
    valid = data.get('validsize') / 10

    X, Y = getXandY(df, xcol, ycol)
    X_scaled = getScaler(X, scaler)
    X_tr, Y_tr, X_te, Y_te, X_va, Y_va = splitData(X_scaled, Y, test, valid)

    trainamount = getAmount(X_tr)
    testamount = getAmount(X_te)
    validamount = getAmount(X_va)
    mdl = modellists.GetModel()

    data['did'] = did
    data['projects'] = prj
    data['columns'] = getColumns(df)
    data['scalerlist'] = mdl.scalerlist
    data['xmemo'] = 'Test_X'
    data['ymemo'] = 'Test_Y'
    data['xfilename'] = filename + '_test_x.csv'
    data['yfilename'] = filename + '_test_y.csv'
    data['xfilesize'] = getFilesize(X_te)
    data['yfilesize'] = getFilesize(Y_te)
    data['xcolumncount'] = len(getColumns(X_te))
    data['ycolumncount'] = len(getColumns(Y_te))
    data['xtest'] = X_te
    data['ytest'] = Y_te
    data['trainamount'] = trainamount
    data['testamount'] = testamount
    data['validamount'] = validamount

@modelselection.route('/setsplit', methods=['PATCH'])
def setsplit():
    """ 데이터셋 분리/스케일러 적용한 데이터를 Temp2에 업데이트 후 모델리스트 리턴 """
    
    data = request.get_json(force=True)
    mdl = modellists.GetModel()
    
    if 'tid' in data:
        tid = data['tid']
        obj = Temp2.query.filter_by(idx=tid).first()
    
    dobj = obj.dataset
    scaler = data['scaler']


    if (instance.testsize == data['testsize']) & (instance.validsize == data['validsize']):
        trainamount = instance.trainamount
        testamount = instance.testamount
        validamount = instance.validamount
    else:
        df = instance.dataset.data
        xcol = data.get('xcol')
        ycol = data.get('ycol')

        test = float(data.get('testsize') / 10)
        valid = float(data.get('validsize') / 10)

        X, Y = getXandY(df, xcol, ycol)
        X_scaled = getScaler(X, scaler)
        X_tr, Y_tr, X_te, Y_te, X_va, Y_va = splitData(X_scaled, Y, test, valid)

        trainamount = getAmount(X_tr)
        testamount = getAmount(X_te)
        validamount = getAmount(X_va)

        data['pid'] = dobj.project.idx
        data['pname'] = dobj.project.pname
        data['dname'] = dobj.dname
        data['did'] = dobj.idx
        data['modellist'] = mdl.modellist
        data['trainamount'] = trainamount
        data['testamount'] = testamount
        data['validamount'] = validamount
        data['filename'] = dobj.filename
    
    response = {
        'tid' : data.get('tid'),
        'pid' : data.get('pid'),
        'did' : data.get('did'),
        'pname' : data.get('pname'),
        'dname' : data.get('dname'),
        'filename' : data.get('filename'),
        'modellist' : data.get('modellist')
        }

    return response


    





