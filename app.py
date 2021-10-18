from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from extension import db
from flask_restplus import Api
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['JSON_SORT_KEYS'] = False
api = Api(app)

# db = SQLAlchemy(app)
db.init_app(app)
migrate = Migrate(app, db)

# import blueprints
from aimodels import aimodels
from datasets import datasets
from modelselection import modelselection
from preprocessing import preprocessing
from projects import projects
from traintest import traintest
from multi import multi
from user import user

SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "A2LAB AI VOUCHER PROJECT API v1"
    }
)
app.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)
app.register_blueprint(aimodels, url_prefix="/aimodels")
app.register_blueprint(datasets, url_prefix="/datasets")
app.register_blueprint(modelselection, url_prefix="/modelselection")
app.register_blueprint(multi, url_prefix="/multi")
app.register_blueprint(preprocessing, url_prefix="/preprocessing")
app.register_blueprint(projects, url_prefix="/projects")
app.register_blueprint(traintest, url_prefix="/traintest")
app.register_blueprint(user, url_prefix="/user")

@app.route("/")
def main():
    return "<h1>Test</h1>"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)