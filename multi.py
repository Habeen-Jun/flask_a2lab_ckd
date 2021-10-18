from flask import Blueprint, render_template, request, jsonify
from extension import db
from core import modellists

multi = Blueprint("multi", __name__)


@multi.route("detail/mtid=<mtid>&model=<model>", methods=["GET"])
def detail(mtid, model):
    print(mtid)
    print(model)
    return jsonify({"success": "love you"})


@multi.route("multiupload/", methods=["POST"])
def multiupload():
    pass



