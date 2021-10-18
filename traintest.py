from flask import Blueprint, render_template, request, jsonify

traintest = Blueprint("traintest", __name__)


@traintest.route("deletetemp/mid=<mid>&tdid=<tdid>", methods=["DELETE"])
def deletetemp(mid, tdid):
    print(mid, tdid)
    return ''
    

@traintest.route("enterresult/<mid>", methods=["GET"])
def enterresult(mid):
    print(mid)
    return 'good'

@traintest.route("entertest/mid=<mid>&tdid=<tdid>", methods=["GET"])
def entertest(mid, tdid):
    print(mid, tdid)
    return 'good'

@traintest.route("entertrain/<mid>", methods=["GET"])
def entertrain(mid):
    print(mid)
    return mid

@traintest.route("entervalid/<mid>", methods=["GET"])
def entervalid(mid):
    return mid
@traintest.route("enterxai/<mid>", methods=["GET"])
def enterxai(mid):
    return mid

@traintest.route("pretest/<mid>", methods=["GET"])
def pretest(mid):
    return mid

@traintest.route("starttest/mid=<mid>&tdid=<tdid>", methods=["GET"])
def starttest(mid, tdid):
    return mid, tdid

@traintest.route("starttrain/", methods=["PATCH"])
def starttrain():
    if not request.json or "mid" not in request.json:
        abort(400)
    return request.json.get("mid")