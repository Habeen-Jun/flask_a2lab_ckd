"""empty message

Revision ID: 49fe0d0e60ce
Revises: c7e8a80e0a36
Create Date: 2021-10-09 15:32:05.923947

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import Text
# revision identifiers, used by Alembic.
revision = '49fe0d0e60ce'
down_revision = 'c7e8a80e0a36'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('data',
    sa.Column('idx', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('project_id', sa.Integer(), nullable=True),
    sa.Column('dname', sa.CHAR(), nullable=False),
    sa.Column('dmemo', sa.CHAR(), nullable=False),
    sa.Column('filename', sa.CHAR(), nullable=False),
    sa.Column('filesize', sa.Integer(), nullable=False),
    sa.Column('columncount', sa.Integer(), nullable=False),
    sa.Column('data', postgresql.JSON(astext_type=Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['project_id'], ['project.idx'], ),
    sa.PrimaryKeyConstraint('idx')
    )
    op.create_table('ai_model',
    sa.Column('idx', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('dataset', sa.Integer(), nullable=True),
    sa.Column('xcol', postgresql.JSON(astext_type=Text()), nullable=False),
    sa.Column('ycol', sa.CHAR(), nullable=False),
    sa.Column('testsize', sa.DECIMAL(), nullable=False),
    sa.Column('validsize', sa.DECIMAL(), nullable=False),
    sa.Column('trainamount', sa.Integer(), nullable=False),
    sa.Column('testamount', sa.Integer(), nullable=False),
    sa.Column('validamount', sa.Integer(), nullable=False),
    sa.Column('scaler', sa.CHAR(), nullable=False),
    sa.Column('params', postgresql.JSON(astext_type=Text()), nullable=False),
    sa.Column('mname', sa.CHAR(), nullable=False),
    sa.Column('mdl_pkl', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['dataset'], ['data.idx'], ),
    sa.PrimaryKeyConstraint('idx')
    )
    op.create_table('temp',
    sa.Column('idx', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('dataset', sa.Integer(), nullable=True),
    sa.Column('data', postgresql.JSON(astext_type=Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['dataset'], ['data.idx'], ),
    sa.PrimaryKeyConstraint('idx')
    )
    op.create_table('temp2',
    sa.Column('idx', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('model', sa.Integer(), nullable=True),
    sa.Column('xcol', postgresql.JSON(astext_type=Text()), nullable=False),
    sa.Column('ycol', sa.CHAR(), nullable=False),
    sa.Column('trainsize', sa.DECIMAL(), nullable=False),
    sa.Column('testsize', sa.DECIMAL(), nullable=False),
    sa.Column('trainsize2', sa.DECIMAL(), nullable=False),
    sa.Column('validsize', sa.DECIMAL(), nullable=False),
    sa.Column('trainamount', sa.Integer(), nullable=False),
    sa.Column('testamount', sa.Integer(), nullable=False),
    sa.Column('validamount', sa.Integer(), nullable=False),
    sa.Column('scaler', sa.CHAR(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['model'], ['data.idx'], ),
    sa.PrimaryKeyConstraint('idx')
    )
    op.create_table('temp3',
    sa.Column('idx', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('model', sa.Integer(), nullable=True),
    sa.Column('testdata', sa.Integer(), nullable=True),
    sa.Column('predict', postgresql.JSON(astext_type=Text()), nullable=False),
    sa.Column('flag', sa.Boolean(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['model'], ['ai_model.idx'], ),
    sa.ForeignKeyConstraint(['testdata'], ['data.idx'], ),
    sa.PrimaryKeyConstraint('idx')
    )
    op.create_table('train_result',
    sa.Column('idx', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('model', sa.Integer(), nullable=True),
    sa.Column('result1', postgresql.JSON(astext_type=Text()), nullable=False),
    sa.Column('result2', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['model'], ['ai_model.idx'], ),
    sa.PrimaryKeyConstraint('idx')
    )
    op.create_table('valid_result',
    sa.Column('idx', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('model', sa.Integer(), nullable=True),
    sa.Column('valid1', postgresql.JSON(astext_type=Text()), nullable=False),
    sa.Column('valid2', postgresql.JSON(astext_type=Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['model'], ['ai_model.idx'], ),
    sa.PrimaryKeyConstraint('idx')
    )
    op.create_table('xai_result',
    sa.Column('idx', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('model', sa.Integer(), nullable=True),
    sa.Column('xai1', postgresql.JSON(astext_type=Text()), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['model'], ['ai_model.idx'], ),
    sa.PrimaryKeyConstraint('idx')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('xai_result')
    op.drop_table('valid_result')
    op.drop_table('train_result')
    op.drop_table('temp3')
    op.drop_table('temp2')
    op.drop_table('temp')
    op.drop_table('ai_model')
    op.drop_table('data')
    # ### end Alembic commands ###