import dotenv, os
from sqlalchemy import create_engine, MetaData, Table, Column, Text, Boolean

dotenv.load_dotenv()
conn_string = os.getenv("DATABASE_URL")

engine = create_engine(conn_string)

metadata = MetaData()
user_table = Table(
    "likes",
    metadata,
    Column('agent_type', Text),
    Column('prompt', Text),
    Column('completion', Text),
    Column('label', Boolean)
)

def insert(agent_type , prompt, completion, label):

    stmt = user_table.insert().values(agent_type=agent_type , prompt=prompt, completion=completion, label=label)
    with engine.connect() as conn:
        _ = conn.execute(stmt)
        conn.commit()
