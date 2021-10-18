from extension import db
import datetime
from sqlalchemy import Column, Integer, Boolean, String, ForeignKey, DECIMAL, DateTime
from sqlalchemy.types import CHAR
from sqlalchemy.dialects.postgresql import JSON

class AIModel(db.Model):
    idx = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset = Column(Integer, ForeignKey('data.idx')) # 데이터 FK
    # data = models.JSONField(null=True)                                            # 모델 적용 시점의 데이터
    xcol = Column(JSON, nullable=False) # X 컬럼명 리스트
    ycol = Column(CHAR, nullable=False)  # Y 컬럼명
    testsize = Column(DECIMAL, nullable=False) # 테스트셋 크기
    validsize = Column(DECIMAL, nullable=False) # 검증셋 크기
    trainamount = Column(Integer, nullable=False) # 훈련셋 데이터수
    testamount = Column(Integer, nullable=False)                                     # 테스트셋 데이터수
    validamount = Column(Integer, nullable=False)                                    # 검증셋 데이터수
    scaler = Column(CHAR, nullable=False)                             # 스케일러
    params = Column(JSON, nullable=False)                                        # 파라미터 설정값
    mname = Column(CHAR, nullable=False)                          # 모델명
    mdl_pkl = Column(String, nullable=False)                          # 모델 PKL 파일 이름
    # user = models.ForeignKey(User, related_name='model', null=True, blank=True, on_delete=models.CASCADE)  # USER FK
    created_at =  Column(DateTime, default=datetime.datetime.utcnow)


    def as_dict(self):
        return {x.name: getattr(self, x.name) for x in self.__table__.columns}

class TrainResult(db.Model):
    idx = db.Column(db.Integer, primary_key=True, autoincrement=True)
    model = Column(Integer, ForeignKey('ai_model.idx')) # 모델 FK
    result1 = Column(JSON, nullable=False) # 피쳐중요도 or traingraph json
    result2 = Column(String, nullable=False) # 트리 이미지파일 경로
    created_at =  Column(DateTime, default=datetime.datetime.utcnow)

    def as_dict(self):
        return {x.name: getattr(self, x.name) for x in self.__table__.columns}


class ValidResult(db.Model):
    idx = db.Column(db.Integer, primary_key=True, autoincrement=True)
    model = Column(Integer, ForeignKey('ai_model.idx')) # 모델 FK
    valid1 = Column(JSON, nullable=False) # 산점도추세선 or confusion matrix json
    valid2 = Column(JSON, nullable=False) # RMSE_MAE표 or ROC curve json
    created_at =  Column(DateTime, default=datetime.datetime.utcnow)

    def as_dict(self):
        return {x.name: getattr(self, x.name) for x in self.__table__.columns}


class XAIResult(db.Model):
    idx = db.Column(db.Integer, primary_key=True, autoincrement=True)
    model = Column(Integer, ForeignKey('ai_model.idx')) # 모델 FK
    xai1 = Column(JSON, nullable=False) # 산점도추세선 or confusion matrix json
    created_at =  Column(DateTime, default=datetime.datetime.utcnow)


    def as_dict(self):
        return {x.name: getattr(self, x.name) for x in self.__table__.columns}


class Temp2(db.Model):
    "model selection "
    idx = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset = Column(Integer, ForeignKey('data.idx')) # 모델 FK
    xcol = Column(JSON, nullable=False)
    ycol = Column(CHAR, nullable=False) 
    trainsize = Column(db.Float, nullable=False)
    testsize = Column(db.Float, nullable=False)
    trainsize2 = Column(db.Float, nullable=False)
    validsize = Column(db.Float, nullable=False)
    trainamount = Column(Integer, nullable=True)
    testamount = Column(Integer, nullable=True)
    validamount = Column(Integer, nullable=True)
    scaler = Column(CHAR, nullable=False)
    created_at =  Column(DateTime, default=datetime.datetime.utcnow)

    def as_dict(self):
        return {x.name: getattr(self, x.name) for x in self.__table__.columns}


class Temp3(db.Model):
    idx = db.Column(db.Integer, primary_key=True, autoincrement=True)
    model = Column(Integer, ForeignKey('ai_model.idx'))
    testdata = Column(Integer, ForeignKey('data.idx'))
    predict = Column(JSON, nullable=False)
    flag = Column(Boolean, nullable=False)
    created_at =  Column(DateTime, default=datetime.datetime.utcnow)

    def as_dict(self):
        return {x.name: getattr(self, x.name) for x in self.__table__.columns}

# projects 
class Project(db.Model):
    idx = db.Column(db.Integer, primary_key=True, autoincrement=True)
    pname = Column(String, nullable=False)
    pmemo = Column(String, nullable=False)
    data = db.relationship('Data', backref='project', uselist=True, lazy=True)
    # user = Column(String, nullable=False)
    created_at =  Column(DateTime, default=datetime.datetime.utcnow)


    def as_dict(self):
        return {x.name: getattr(self, x.name) for x in self.__table__.columns}

# datasets
class Data(db.Model):
    idx = db.Column(db.Integer, primary_key=True, autoincrement=True)
    project_id = Column(Integer, ForeignKey('project.idx')) #임시 주석... 
    dname = Column(CHAR, nullable=False)
    dmemo = Column(CHAR, nullable=False)
    filename = Column(CHAR, nullable=False)
    filesize = Column(Integer, nullable=False)
    columncount = Column(Integer, nullable=False)
    data = Column(JSON, nullable=False)
    created_at =  Column(DateTime, default=datetime.datetime.utcnow)

    def as_dict(self):
        return {x.name: getattr(self, x.name) for x in self.__table__.columns}


class Temp(db.Model):
    "학습 데이터 preprocessing"
    idx = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dataset = Column(Integer, ForeignKey('data.idx'))
    data = Column(JSON, nullable=False)
    created_at =  Column(DateTime, default=datetime.datetime.utcnow)


    def as_dict(self):
        return {x.name: getattr(self, x.name) for x in self.__table__.columns}






                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      