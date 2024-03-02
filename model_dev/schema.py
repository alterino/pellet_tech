import sqlalchemy as db
import sqlalchemy.orm as orm

from sqlalchemy.ext.declarative import declarative_base

Base: orm.DeclarativeBase = declarative_base()

class Device(Base):
    """
    Represents a device.

    Attributes:
        id (int): The primary key for the device.
        name (str): The name of the device.
        process_data (relationship): Relationship with ProcessData table.
        process_predictions (relationship): Relationship with ProcessPredictions table.
        events (relationship): Relationship with Events table.
        state (relationship): Relationship with State table.
    """

    __tablename__ = 'device'
    
    id: int = db.Column(db.Integer, primary_key=True)
    name: str = db.Column(db.Text)
    process_data = orm.relationship('ProcessData', back_populates='device')
    process_predictions = orm.relationship('ProcessPredictions', back_populates='device')
    events = orm.relationship('Events', back_populates='device')


class ProcessData(Base):
    """
    Represents processed data from a device.

    Attributes:
        id (int): The primary key for the process data.
        device_id (int): Foreign key referencing the device.
        date (DateTime): Timestamp for the process data.
        fault_code (str): Fault code associated with the process data.
    """

    __tablename__ = 'process_data'
    
    id: int = db.Column(db.Integer, primary_key=True)
    device_id: int = db.Column(db.Integer, db.ForeignKey('device.id'))
    date: db.DateTime = db.Column(db.DateTime)
    fault_code: str = db.Column(db.Text)
    device = orm.relationship('Device', back_populates='process_data')


class ProcessPredictions(Base):
    """
    Represents model predictions for a device.

    Attributes:
        id (int): The primary key for the prediction.
        device_id (int): Foreign key referencing the device.
        model (str): The model used for prediction.
        date (DateTime): Timestamp for the prediction.
        fail_probability (float): Probability of failure.
        fail_time (DateTime): Predicted failure time.
    """

    __tablename__ = 'process_predictions'
    
    id: int = db.Column(db.Integer, primary_key=True)
    device_id: int = db.Column(db.Integer, db.ForeignKey('device.id'))
    model: str = db.Column(db.Text)
    date: db.DateTime = db.Column(db.DateTime)
    fail_probability: float = db.Column(db.Float)
    fail_time: db.DateTime = db.Column(db.DateTime)
    device = orm.relationship('Device', back_populates='process_predictions')


class Events(Base):
    """
    Represents event notifications based on model predictions.

    Attributes:
        id (int): The primary key for the event.
        device_id (int): Foreign key referencing the device.
    """

    __tablename__ = 'events'
    
    id: int = db.Column(db.Integer, primary_key=True)
    device_id: int = db.Column(db.Integer, db.ForeignKey('device.id'))
    prediction_id: int = db.Column(db.Integer, db.ForeignKey('process_predictions.id'))
    create_date: db.DateTime = db.Column(db.DateTime)
    last_updated: db.DateTime = db.Column(db.DateTime)
    status: str = db.Column(db.Text)
    severity: str = db.Column(db.Text)
    fail_probability: float = db.Column(db.Float)
    device = orm.relationship('Device', back_populates='events')


class State(Base):
    """
    Represents the current state of ETL/Model/Update runs of the app.

    Attributes:
        name (str): The primary key for the state.
        device_id (int): Foreign key referencing the device.
        last_id (int): The last processed ID.
    """

    __tablename__ = 'state'
    
    id: int = db.Column(db.Integer, primary_key=True)
    name: str = db.Column(db.Text)
    device_id: int = db.Column(db.Integer, default=0)
    last_id: int = db.Column(db.Integer, default=0)


def get_metadata() -> db.MetaData:
    """ Retrieve the ORM database MetaData object
    """
    return Base.metadata


def create_tables(engine: db.Engine):
    """ Create all tables from ORM
    """
    get_metadata().create_all(engine)

def drop_tables(engine: db.Engine):
    get_metadata().drop_all(engine)
