from flask import Blueprint, render_template, jsonify
from models import AIModel, Data
from extension import db

aimodels = Blueprint("aimodels", __name__)

@aimodels.route("")
def home():
    return "<h1>aimodels home</h1>"

@aimodels.route("deleteallmodel/<data_id>", methods=['DELETE'])
def deleteallmodel(data_id):
    """
    해당 데이터셋에 대한 모델(전체)를 삭제한다
    """
    models_to_delete = AIModel.query.filter_by(dataset=data_id).all()

    if len(models_to_delete) == 0:
        return jsonify({"message": "no model found"})

    project_id = models_to_delete[0].dataset.project.idx
    dataset_id = data_id

    #for all records
    db.session.query(models_to_delete).delete()
    db.session.commit()
    
    return jsonify({
        'pid': project_id,
        'did': dataset_id,
        'message': "Success"
    })

@aimodels.route("deletemodel/<model_id>", methods=['DELETE'])
def deletemodel(model_id):
    """
    모델(하나)를 삭제한다
    """
    model_to_delete = AIModel.query.filter_by(idx=model_id).first()
    
    if model_to_delete == None:
        return jsonify({"message": "no model found"})

    project_id = model_to_delete.dataset.project.idx
    dataset_id = model_to_delete.dataset.idx
    model_id = model_to_delete.idx

    db.session.delete(model_to_delete)
    db.session.commit()

    return jsonify({
        'project_id': project_id,
        'dataset_id': dataset_id,
        'model_id': model_id,
        'message': 'Success'
    })

@aimodels.route("modellist/<data_id>")
def modellist(data_id):
    """
    데이터 id를 전달받아 해당 데이터에 포함된 모델 리스트를 반환
    """
    models = AIModel.query.filter_by(dataset=data_id).all()
    print(models)
    if models == []:
        return jsonify({"message": "no model found"})
    
    return jsonify([d.as_dict() for d in models])