from flask import Blueprint, render_template, jsonify, request, flash, redirect
from models import AIModel, Data, Project
from extension import db
from core import modellists
from core.datafix import getCorr, getColumns, fileToJson, getDescribe
from core.modeltrain import *
from flask import make_response
from werkzeug.utils import secure_filename
import os 
 

datasets = Blueprint("datasets", __name__)

UPLOAD_FOLDER = os.getcwd()+'\\tmp\\uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv', 'hwp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_fields(request, required_fields):
    for field in required_fields:
        field_value = request.form.get(field) 
        if field_value == '':
            empty_field = field
            return empty_field
    return None

@datasets.route("/")
def dataset_home():
    return jsonify('this is a homepage of datasets')

@datasets.route('/uploadfile', methods=['POST'])
def uploadfile():
    required_fields = ['pid', 'dname', 'dmemo']
    if 'dataset' not in request.files:
        return jsonify({"msg":"file field is required."})

    pid = request.form.get('pid')#json.loads(request.form.get('pid'))
    dname = request.form.get('dname')#json.loads(request.form.get('dname'))
    dmemo = request.form.get('dmemo')#json.loads(request.form.get('dmemo'))

    project = Project.query.filter_by(idx=pid).first()
    if project is None:
        return {
            "message": "project does not exist."
        }
    
    # field check
    empty_value = check_fields(request, required_fields)
    if empty_value is not None:
        print(f'empty_value: {empty_value}')
        return jsonify({'required field missing': f'required field "{empty_value}" is missing'})

       
    file = request.files['dataset']
    
    
    if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # 09/27 file.save 로 메모리 상 데이터 사라졌던것.
        # file.save(os.path.join(UPLOAD_FOLDER, filename))

    # 파일사이즈 추가
    size = file.getbuffer().nbytes
    unit = 'Bytes'
    if size > 1024:
        size = round(size / 1024, 2)
        unit = 'KB'
        if size > 1024:
            size = round(size / 1024, 2)
            unit = 'MB'

    size = str(size) + ' ' + unit




    # 데이터 df로 변환 & json으로 변환 & describe, corr, column수 리턴
    df, desc, corr, cols = fileToJson(file)
    # wtf? 10/09

    record = Data(
        project_id=int(pid),
        dname=dname,
        dmemo=dmemo,
        filename=filename,
        filesize=size,
        columncount=cols,
        data=df,
    )    
    
    print(df)

    db.session.add(record)
    db.session.commit()

    return jsonify({
        "did": record.idx,
        "pid": pid,
        "message": "Success"
    })

@datasets.route('datalist/<project_id>')
def datalist(project_id):
    """
    프로젝트 id를 전달받아 해당 프로젝트에 포함된 데이터 리스트를 반환
    """
    datasets = Data.query.filter_by(project_id=project_id).all()
    project = Project.query.filter_by(idx=project_id).first()
    
    if project is None:
        return {
            "message": "project does not exist."
        }
    
    if datasets == []:
        response = {
            'msg': 'project not found'
        }
        return jsonify(response)

    response = {
        "pid": project_id,
        "pname": project.pname,
        "datalist": {}
    }

    for idx, data in enumerate(datasets):
        instance = {
                    'did': data.idx,
                    'dname': data.dname,
                    'dmemo': data.dmemo,
                    'filename': data.filename,
                    'filesize': data.filesize,
                    'columncount': data.columncount,
                    'modelcount': len(datasets),
                    'created_at': data.created_at,
                }
        response["datalist"][str(idx+1)] = instance

    return jsonify(response)
    
@datasets.route('editdata/', methods=['PATCH'])
def editdata():
    """
    데이터 id를 전달받아 해당 데이터 정보 수정
    """
    data = request.get_json(force=True)

    data_id = data['did']

    query = Data.query.filter_by(idx=data_id)
    data_to_edit = query.first()
    
    if data_to_edit == None:
        return jsonify({"message": "dataset not found"})    
    
    data_to_edit.dname = data['dname']
    data_to_edit.dmemo = data['dmemo']
    
    db.session.commit()

    return jsonify({
        "pid": data_to_edit.project_id,
        "did": data_to_edit.idx,
    })

@datasets.route('/duplicatedata', methods=["POST"])
def duplicatedata():
    """
    신규 파일을 업로드한다
    """
    data = request.get_json(force=True)

    dataset = Data.query.filter_by(idx=data['did']).first()

    if dataset == None:
        return jsonify({
            "message": "Model not found"
        })
    
    dataset.dname = data['dname']
    dataset.dmemo = data['dmemo']
    
    file_instance = Data(
        project = dataset.project_id,
        dname = data['dname'],
        dmemo = data['dmemo'],
        filesize = dataset.filesize,
        filename = dataset.filename,
        columncount = dataset.columncount,
        data = dataset.data
    )

    db.session.add(file_instance)
    db.session.commit()

    response = {
        "did": file_instance.idx,
        "pid": file_instance.project_id,
        "dname": file_instance.dname,
        "dmemo": file_instance.dmemo,
        "message": "Success"
        }

    return response

@datasets.route('deletedata/<data_id>', methods=["DELETE"])
def deletedata(data_id):
    """
    데이터(하나)를 삭제한다
    """
    query = Data.query.filter_by(idx=data_id)
    dataset_to_delete = query.first()
    

    if dataset_to_delete == None:
        return jsonify({"message": "dataset not found"})
    
    project_id = dataset_to_delete.project_id

    query.delete()
    db.session.commit()

    message = {
        "pid": project_id,
        "did": data_id,
        "message": "Success"
    }

    return make_response(jsonify(message), 200)

@datasets.route('deletealldata/<project_id>', methods=["DELETE"])
def deletealldata(project_id):
    """
    데이터(전체)를 삭제한다
    """
    query = Data.query.filter_by(project_id=project_id)
    datasets_to_delete = query.all()

    if datasets_to_delete == []:
        return jsonify({"message": "dataset not found"})

    query.delete()
    db.session.commit()


    return make_response(jsonify({
        "pid": project_id,
        "message": 'Success'
    }), 200) # recode = 200

@datasets.route("getdetail1/<data_id>", methods=["GET"])
def getdetail1(data_id):
    """
    데이터 개별 정보 조회 - 요약정보
    """

    dataset = Data.query.filter_by(idx=data_id.replace('\u200b', '')).first()

    if dataset == None:
        return jsonify({"message":"dataset not found"})

    desc = getDescribe(dataset.data)
    raw = dataset.data

    project = Project.query.filter_by(idx=dataset.idx).first()

    response = {
                'did': dataset.idx,
                'pid': dataset.project_id,
                'pname': project.pname,
                'dname': dataset.dname,
                'filename': dataset.filename,
                'data': raw,
                'describe': desc,
            }
    
    return jsonify(response)

@datasets.route("getdetail2/<data_id>")
def getdetail2(data_id):
    """
    데이터 개별 정보 조회 - 그래프
    """
    dataset = Data.query.filter_by(idx=data_id.replace('\u200b', '')).first()

    if dataset == None:
        return jsonify({"message":"dataset not found"})

    raw = dataset.data
    corr = getCorr(dataset.data)
    cols = getColumns(dataset.data)

    project = Project.query.filter_by(idx=dataset.idx).first()


    response = {
        'did': dataset.idx,
        'pid': dataset.project_id,
        'pname': project.pname,
        'dname': dataset.dname,
        'filename': dataset.filename,
        'data': raw,
        'corr': corr,
        'columns': cols,
    }

    return jsonify(response)
    
@datasets.route("getdetail3/<data_id>")
def getdetail3(data_id):
    """
    데이터 개별 정보 조회 - 원본데이터
    """
    dataset = Data.query.filter_by(idx=data_id).first()
    if dataset == None:
        return jsonify({"message":"dataset not found"})

    return jsonify(dataset.as_dict())



