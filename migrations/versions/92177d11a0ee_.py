"""empty message

Revision ID: 92177d11a0ee
Revises: 49fe0d0e60ce
Create Date: 2021-10-15 10:58:58.286527

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '92177d11a0ee'
down_revision = '49fe0d0e60ce'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('temp2', sa.Column('dataset', sa.Integer(), nullable=True))
    op.drop_constraint(None, 'temp2', type_='foreignkey')
    op.create_foreign_key(None, 'temp2', 'data', ['dataset'], ['idx'])
    op.drop_column('temp2', 'model')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('temp2', sa.Column('model', sa.INTEGER(), nullable=True))
    op.drop_constraint(None, 'temp2', type_='foreignkey')
    op.create_foreign_key(None, 'temp2', 'data', ['model'], ['idx'])
    op.drop_column('temp2', 'dataset')
    # ### end Alembic commands ###
