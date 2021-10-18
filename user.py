from flask import Blueprint, render_template, jsonify, request, abort
from models import Project
from extension import db
import json


user = Blueprint("user", __name__)

@user.route("createuser/", methods=["POST"])
def createuser():
    if not request.json or "userid" not in request.json:
        abort(400)
    return jsonify({"message": "success"})

