"""empty message

Revision ID: 7493263ee44a
Revises: fb540faec117
Create Date: 2021-10-15 11:01:20.639071

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.types import Text
# revision identifiers, used by Alembic.
revision = '7493263ee44a'
down_revision = 'fb540faec117'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('temp2',
    sa.Column('idx', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('dataset', sa.Integer(), nullable=True),
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
    sa.ForeignKeyConstraint(['dataset'], ['data.idx'], ),
    sa.PrimaryKeyConstraint('idx')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('temp2')
    # ### end Alembic commands ###