from flask import Blueprint, render_template, jsonify, request
from extension import db
from core import modellists
from models import Temp, Data, Project
from core.datafix import *
preprocessing = Blueprint("preprocessing", __name__)

@preprocessing.route("deletetemp/<tid>", methods=["DELETE"])
def deletetemp(tid):
    data = Temp.query.filter_by(idx=tid).first()

    if data is None:
        return jsonify({
            "message": "temp does not exist"
        })
    
    #for all records
    db.session.delete(data)
    db.session.commit()

    did = data.dataset
    dobj = Data.query.filter_by(idx=did).first()
    if dobj is None:
        return {
            "message": "data does not exist."
        }
    pid = dobj.project.idx

    message = {'pid':pid,
               'did':did,
               'message':"Success"}

    return jsonify(message)

@preprocessing.route("preencoder/<did>", methods=['GET'])
def preencoder(did): 
    """
    인코더 처리 페이지 진입 (10/6 테스트 완료)
    """
    data_record = Data.query.filter_by(idx=did).first()
    if data_record is None:
        return jsonify({
            "message": "dataset does not exist"
        })
    mdl = modellists.GetModel()
    columns = getColumns(data_record.data)
    head = getHead(data_record.data)
    encodelist = mdl.encoderlist

    if data_record == None:
        response = {
                'message' : 'Data not found'
        }
        return jsonify(response)

    # linked project with data record
    linked_project = Project.query.filter_by(idx=data_record.project_id).first()

    if linked_project is None:
        response = {
            "message": "Linked project is not found."
        }
        return jsonify(response)

    instance = {
        'pid': data_record.project_id,
        'did': data_record.idx,
        'pname': linked_project.pname,
        'dname': data_record.dname,
        'filename': data_record.filename,
        'head': head,
        'columns': columns,
        'encoderlist': encodelist,
    }

    return jsonify(instance)

@preprocessing.route("premissing/<did>", methods=['GET'])
def premissing(did):
    """
    결측치 처리 페이지 진입 (10/6 테스트 완료)
    """
    data_record = Data.query.filter_by(idx=did).first()
    if data_record is None:
        return jsonify({
            "message": "dataset does not exist"
        })
    mdl = modellists.GetModel()
    columns = getColumns(data_record.data)
    graph = getNans(data_record.data)
    mvalues = mdl.ifnanlist

    # linked project with data record
    linked_project = Project.query.filter_by(idx=data_record.project_id).first()

    if linked_project is None:
        response = {
            "message": "Linked project is not found."
        }
        return jsonify(response)

    instance = {
        'pid': data_record.project_id,
        'did': data_record.idx,
        'pname': linked_project.pname,
        'dname': data_record.dname,
        'filename': data_record.filename,
        'graph': graph,
        'columns': columns,
        'mvalues': mvalues,
    }
    
    print(data_record.data)
    
    return jsonify(instance)

@preprocessing.route("preoutlier/<did>", methods=['GET'])
def preoutlier(did):
    """
    이상치 처리 페이지 진입 (10/6 테스트 완료)
    """

    data_record = Data.query.filter_by(idx=did).first()
    if data_record is None:
        return jsonify({
            "message": "dataset does not exist"
        })
    mdl = modellists.GetModel()
    columns = getColumns(data_record.data)
    raw = json.loads(data_record.data)

    # linked project with data record
    linked_project = Project.query.filter_by(idx=data_record.project_id).first()

    if linked_project is None:
        response = {
            "message": "Linked project is not found."
        }
        return jsonify(response)

    instance = {
        'pid': data_record.project_id,
        'did': data_record.idx,
        'pname': linked_project.pname,
        'dname': data_record.dname,
        'filename': data_record.filename,
        'columns': columns,
        'datapick': {},
        'data': raw
    }

    return jsonify(instance)

@preprocessing.route("/saveencoder", methods=["patch"])
def saveencoder():
    """10/15 test completed"""
    # incoming data
    data  = request.get_json(force=True)['data']
    
    # get tid from incoming data
    tid = data['tid']
    temp = Temp.query.filter_by(idx=tid).first()

    if temp is None:
        return {
            "message": "temp does not exist."
        }
    
    
    did = temp.dataset
    dobj = Data.query.filter_by(idx=did).first()

    if did is Nonde:
        return {
            "message": "data does not exist."
        }
    
    mdl = modellists.GetModel()
    df = temp.data
    cols = getColumns(df)
    filesize = getFilesize(df)


    # update data
    dobj.data = temp.data
    dobj.columncount = len(cols)
    dobj.filesize = filesize


    #delete temp
    db.session.delete(temp)
    db.session.commit()

    response = {
        'pid': dobj.project_id,
        'did': dobj.idx,
        'pname': dobj.project.pname,
        'dname': dobj.dname,
        'filename': dobj.filename,
        'columns': cols,
        'head': getHead(df),
        'encoderlist': mdl.encoderlist
    }
    
    return response

@preprocessing.route("savemissing/", methods=["patch"])
def savemissing():
    """10/15 테스트 완료 """
    data  = request.get_json(force=True)['data']
    tid = data['tid']
    mdl = modellists.GetModel()
    temp = Temp.query.filter_by(idx=tid).first()
    if temp is None:
        return jsonify({
            "message": "temp does not exist"
        })
    did = temp.dataset
    df = temp.data

    dobj = Data.query.filter_by(idx=did).first()
    if dobj is None:
        return jsonify({
            "message": "dataset does not exist"
        })
    
    pid = dobj.project_id
    # Data.query.filter_by(project_id=)
    project = Project.query.filter_by(idx=pid).first()

    if project is None:
        return jsonify({
            "message": "project does not exist"
        })    
    
    # 10/09 ValueError: Expected object or value
    cols = getColumns(df)
    # filesize = getFilesize(df)
    graph = getNans(df)
    mvalue = mdl.ifnanlist
    filesize = getFilesize(df)

    if dobj:
        # temp data to already existed Data instance
        dobj.data = temp.data
        dobj.columncount = len(cols)
        dobj.filesize = filesize

        #delete temp
        db.session.delete(temp)
        db.session.commit()
        print(f'temp {tid} deleted')


        response = {
            'pid' : dobj.project_id,
            'did' : dobj.idx,
            'pname' : dobj.project.pname,
            'dname' : dobj.dname,
            'filename' : dobj.filename,
            'columns': cols,
            'graph' : graph,
            'mvalues' : mvalue,
            }

        return response

@preprocessing.route("saveoutlier/", methods=["patch"])
def saveoutlier():
    """ 10/15 test 완료 """
    data  = request.get_json(force=True)['data']
    tid = None
    
    if 'tid' in data:
        tid = data['tid']
        temp = Temp.query.filter_by(idx=tid).first()
        if temp is None:
            return {
                "message": "temp does not exist."
            }
        did = temp.dataset
    else:
        return {
            "message": "tid is required."
        }

    # mdl = modellists.GetModel()

    
    df = temp.data
    print(df)
    cols = getColumns(df)
    filesize = getFilesize(df)
    
    dobj = Data.query.filter_by(idx=did).first()

    project = Project.query.filter_by(idx=dobj.project_id).first()

    # update on dataset
    dobj.data = df
    dobj.columncount = len(cols)
    dobj.filesize = filesize
    db.session.commit()


    # delete temp
    db.session.delete(temp)
    db.session.commit()

    response = {
        "pid": project.idx,
        "did": did,
        "pname": project.pname,
        "dname": dobj.dname,
        "filename": dobj.filename,
        "filesize": filesize,
        "columns": cols,
        "datapick": {},
        "data": df,
    }


    return response
    
@preprocessing.route("searchoutlier/did=<did>&col=<col>&val=<val>")
def searchoutlier(did, col, val):
    """10/15 테스트 완료"""
    dobj = Data.query.filter_by(idx=did).first()
    if dobj is None:
        return {
            "message": "data does not exist."
        }
        
    columns = getColumns(dobj.data)
    datapick = searchOut(dobj.data, col, val)
    raw = json.loads(dobj.data)


    response = {
                'pname': dobj.project.pname,
                'dname': dobj.dname,
                'filename': dobj.filename,
                'data': raw,
                'columns': columns,
                'datapick': datapick,
                'did': dobj.idx,
                'pid': dobj.project.idx
            }
            
    return response

@preprocessing.route("searchoutlier/did=<did>&tid=<tid>&col=<col>&val=<val>")
def searchoutlier2(did, tid, col, val):
    """10/15 테스트 완료"""
    temp = Temp.query.filter_by(idx=tid).first()
    if temp is None:
        return {
            "message": "temp does not exist."
        }
    dobj = Data.query.filter_by(idx=temp.dataset).first()
    if dobj is None:
        return {
            "message": "data does not exist."
        }

    columns = getColumns(temp.data)
    datapick = searchOut(temp.data, col, val)
    raw = json.loads(temp.data)


    response = {
                'tid': temp.idx,
                'pname': dobj.project.pname,
                'dname': dobj.dname,
                'filename': dobj.filename,
                'data': raw,
                'columns': columns,
                'datapick': datapick,
                'did': dobj.idx,
                'pid': dobj.project.idx
            }
            

    return response

@preprocessing.route("setencoder", methods=["patch"])
def setencoder():
    """10/15 test completed"""
    # get temp or create temp
    mdl = modellists.GetModel()
    data  = request.get_json(force=True)
    tid = None
    did = None

    if 'tid' in data:
        tid = data['tid']
    if 'did' in data:
        did = data['did']


    dobj = Data.query.filter_by(idx=did).first()
    if dobj is None:
        return ({
            "message": "dataset does not exist"
        })

    istemp = Temp.query.filter_by(dataset=did).first()

    if tid is not None:
        instance = Temp.query.filter_by(idx=tid).first()
        if instance is None:
            return jsonify({
                "message": "tid does not exist"
            })
    elif istemp is not None:
        instance = Temp.query.filter_by(dataset=did).first()
    else:
        print('create temp')
        new = Temp(
            dataset=did,
            data=dobj.data
        )

        db.session.add(new)
        db.session.commit()
        db.session.refresh(new)
        tid = new.idx
        print(tid)
        
        instance = Temp.query.filter_by(idx=tid).first()


    # perform update
    # print(instance.data)
    encoder = data['encoder']
    col = data['col']

    # print(instance.data)
    # df = getEncoder(instance.data, encoder, col)
    # print(df[1000000])
    try:
        df = getEncoder(instance.data, encoder, col)
    except Exception as e:
        print(repr(e))
        if "Input contains NaN" in str(e):
            return jsonify({
                "error": "결측치 처리를 먼저 진행해주세요"
            })

        return jsonify({
            "error": str(e)
        })
    
    cols = getColumns(df)
    project = Project.query.filter_by(idx=dobj.project_id).first()
    
    # update
    instance.data = df
    db.session.commit()

    response = {
        'tid' : instance.idx,
        'pid' : project.idx,
        'did' : dobj.idx,
        'pname': project.pname,
        'dname': dobj.dname,
        'filename' : dobj.filename,
        "data" : df,
        "columns": cols,
        "head": getHead(df),
        "encoderlist": mdl.encoderlist,
    }

    return response
    
@preprocessing.route("setmissing", methods=["patch"])
def setmissing():
    """ 10/09 테스트 완료"""
    # get temp or create temp
    mdl = modellists.GetModel()
    data  = request.get_json(force=True)

    tid = None
    did = None
    if 'tid' in data:
        tid = data['tid']
    if 'did' in data:
        did = data['did']


    dobj = Data.query.filter_by(idx=did).first()

    if dobj is None:
        return jsonify({
            "message": "dataset does not exist"
        })

    istemp = Temp.query.filter_by(dataset=did).first()

    if tid is not None:
        instance = Temp.query.filter_by(idx=tid).first()
        if instance is None:
            return jsonify({
                "message": "tid does not exist"
            })
    elif istemp is not None:
        instance = Temp.query.filter_by(dataset=did).first()
        tid = instance.idx
    else:
        print('create new')
        new = Temp(
            dataset=did,
            data=dobj.data
        )
        db.session.add(new)
        db.session.commit()
        db.session.refresh(new)
        tid = new.idx

        instance = Temp.query.filter_by(idx=tid).first()


    # perform update
    ifnan = data['ifnan']
    col = data['col']
    print(type(instance.data))
    print(instance.data)
    
    
    try:
        df = missingFix(str(instance.data), ifnan, col)
    except Exception as e:
        return jsonify({
            "error": repr(e)
        })

    col = getColumns(df)
    graph = getNans(df)
    mvalue = mdl.ifnanlist
    pid = dobj.project_id
    dname = dobj.dname
    filename = dobj.filename
    project = Project.query.filter_by(idx=pid).first()

    if project is None:
        return jsonify({
            "message": "project does not exist"
        })

    pname = project.pname

    db.session.add(instance)
    db.session.commit()

    response = {
        "pid": pid,
        "did": did,
        "tid": tid,
        "pname": pname,
        "dname": dname,
        "filename": filename,
        "columns": col,
        "graph": graph,
        "mvalues": mvalue,
        "explain": "결측치란 비어있는 값을 의미한다. 결측치 처리는 크게 입력과 삭제로 나뉜다. 입력은 특정값, 이전값, 이후값, 중간값, 평균값 입력이 있으며, 삭제는 셀, 행 삭제가 있다."
    }

    return jsonify(response)

@preprocessing.route("setoutlier", methods=["patch"])
def setoutlier():
    """10/15 테스트 완료"""
    data  = request.get_json(force=True)
    tid = None
    did = None
    if 'tid' in data:
        tid = data['tid']
    if 'did' in data:
        did = data['did']

    dobj = Data.query.filter_by(idx=did).first()
    if dobj is None:
        return jsonify({
            "message": "dataset does not exist"
        })

    istemp = Temp.query.filter_by(dataset=did).first()

    if tid is not None:
        instance = Temp.query.filter_by(idx=tid).first()
        if instance is None:
            return jsonify({
                "message": "tid does not exist"
            })
    elif istemp is not None:
        instance = Temp.query.filter_by(dataset=did).first()
    else:
        print('create new')
        new = Temp(
            dataset=did,
            data=dobj.data
        )
        db.session.add(new)
        db.session.commit()
        db.session.refresh(new)
        tid = new.idx

        instance = Temp.query.filter_by(idx=tid).first()

    data1 = str(instance.data)

    datapick = data.get('datapick')
    df, df_cut = setOut(data1, datapick)
    columns = getColumns(df)
    
    response = {
        'tid' : instance.idx,
        'pid' : dobj.idx,
        'did' : dobj.idx,
        'pname' : dobj.project.pname,
        'dname' : dobj.dname,
        'filename' : dobj.filename,
        'columns': columns,
        'datapick': datapick,
        'data': dobj.data,
        }
    

    return response 


