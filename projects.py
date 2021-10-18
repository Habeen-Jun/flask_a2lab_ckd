from flask import Blueprint, render_template, jsonify, request
from models import Project
from extension import db
import json


projects = Blueprint("projects", __name__)


@projects.route('/createproject', methods=["POST"])
def create_project():
    """ """
    data = request.get_json(force=True)

    pname = data["pname"]
    pmemo = data["pmemo"]

    # pname 중복 시 
    duplicated_data = Project.query.filter_by(pname=pname).first()
    if duplicated_data:
        return jsonify({
            "pname": [
                "Name already exists"
            ]
        })
    
    data = Project(pname=pname, pmemo=pmemo)

    db.session.add(data)
    db.session.commit()

    pid = Project.query.filter_by(pname=pname).first().idx
    
    return jsonify({
        "pid": pid,
        "message": "success"
    })


@projects.route('/deleteproject/<project_id>', methods=["DELETE"])
def delete_project(project_id):
    query = Project.query.filter_by(idx=project_id)
    project_to_delete = query.first()

    print(project_to_delete)
    if project_to_delete is None:
        return jsonify({
            "message": "project does not exist."
        })
    
    # 삭제
    query.delete()
    db.session.commit()

    msg = {
        'pid': project_id,
        'message': 'Success'
    }
    return jsonify(msg)


@projects.route('/editproject', methods=["PATCH"])
def edit_project():
    data = request.get_json(force=True)

    pid = data["pid"]
    pname = data["pname"]
    pmemo = data["pmemo"]

    print(pid, pname, pmemo)
    


    query = Project.query.filter_by(idx=pid)
    project_to_edit = query.first()
    
    if project_to_edit is None:
        return jsonify({
            "message": "project does not exist."
        })

    # pname 중복 시 
    duplicated_data = Project.query.filter_by(pname=pname).first()
    print(duplicated_data)
    if duplicated_data:
        return jsonify({
            "non_field_errors" : [
                "name already exists"
            ]
        })

    project_to_edit.pname = pname
    project_to_edit.pmemo = pmemo

    db.session.commit()

    return jsonify({
        "pid" : pid,
        "message": "success"
    })

    


@projects.route('/projectlist', methods=["GET"])
def project_list():
    projects = Project.query.all()

    response = {}


    for idx, record in enumerate(projects):
        print(record.data)
        data = {
            "pid": record.idx,
            "pname": record.pname,
            "pmemo": record.pmemo,
            "amount": len(record.data),#.all()
            "created_at": record.created_at
        }
        response[idx+1] = data

    
    return response

