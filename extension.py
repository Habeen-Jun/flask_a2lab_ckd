from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
# circular import 막기 위해 db app.py 에서 분리
db = SQLAlchemy()
