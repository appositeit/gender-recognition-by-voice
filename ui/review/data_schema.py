import sqlalchemy as db
from sqlalchemy.orm import Session
'''Utility functions for storing and using structured data.'''


class GenderDB:

    def __init__(self):
        #  self.connection = self.connect(check_same_thread=False)
        self.connection = self.connect()
        self.metadata = db.MetaData()
        self.metadata.reflect(self.engine)

        if 'originals' not in self.metadata.tables.keys():
            print('Creating tables')
            self.create_tables()
        self.get_tables()

    def get_tables(self):
        self.originals = db.Table('originals', self.metadata, autoload_with=self.engine)
        self.samples = db.Table('samples', self.metadata, autoload_with=self.engine)
        self.tests = db.Table('tests', self.metadata, autoload_with=self.engine)


    def connect(self):
        '''Initiate connection to the database.'''
        # engine = create_engine("mysql://gender_recognition:@localhost/gr")
        self.engine = db.create_engine('sqlite:////mnt/fastest/jem/ml/gender-recognition-by-voice/genderml.sqlite')
        return self.engine.connect()


    def create_tables(self):
        '''Creates the database schema.'''

        originals = db.Table('originals', self.metadata,
                      db.Column('id', db.Integer(), db.Identity(), primary_key=True),
                      db.Column('filename', db.String(255), nullable=False, index=True, unique=True),
                      db.Column('male', db.Boolean(), default=True),
                      db.Column('frequency', db.Float(), default=0),
                      db.Column('confidence', db.Float(), default=0),
                      db.Column('length', db.Float(), default=0),
                      )

        samples = db.Table('samples', self.metadata,
                      db.Column('id', db.Integer(), db.Identity(), primary_key=True),
                      db.Column('filename', db.String(255), nullable=False, index=True, unique=True),
                      db.Column('original_id', db.ForeignKey('originals.id')),
                      db.Column('parent_offset', db.Float(), default=0),
                      db.Column('male', db.Boolean(), default=True),
                      db.Column('frequency', db.Float(), default=0),
                      db.Column('confidence', db.Float(), default=0),
                      )

        test = db.Table('tests', self.metadata,
                      db.Column('id', db.Integer(), db.Identity()),
                      db.Column('test_run', db.Integer, nullable=False),
                      db.Column('sample_id', db.ForeignKey('samples.id')),
                      db.Column('male', db.Float(), default=1.0)
                      )
        self.metadata.create_all(self.engine)  # Creates the table


    def drop_tables(self):
        '''Clear the tables from the database.'''
        db.metadata.drop_all()



    def add_originals(self, data):
        try:
            query = db.insert(self.originals)
            result = self.connection.execute(query, data)
        except db.exc.IntegrityError:
            return None
        return result.inserted_primary_key[0]

    def read_original(self, filename):
        query = db.select([self.originals]).where(self.originals.columns.filename == filename)
        return self.connection.execute(query).all()

    def update_original(self, id, values):
        query = self.originals.update().values(**values).where(self.originals.c.id == id)
        return self.connection.execute(query)

    def add_samples(self, data):
        '''Add an array of file entries to the samples table.

        Arguments:
            data: A list of sample dicts to add to samples table.
        '''
        try:
            query = db.insert(self.samples)
            _ = self.connection.execute(query, data)
        except db.exc.IntegrityError:
            return None

    def read_samples(self, sort=None, order='asc', page=0):
        query = db.select([self.samples])
        if sort:
            sort_string = sort
            query = db.select([self.samples]).order_by(
                    getattr(getattr(self.samples.c, sort), order)())
        else:
            query = db.select([self.samples])

        return self.connection.execute(query).mappings().all()

    def update_sample(self, id, values):
        query = self.samples.update().values(**values).where(self.samples.c.id == id)
        return self.connection.execute(query)


    def reset_tables(self):
        connect()
        drop_tables()
        create_tables()


def main():
    _ = GenderDB()
if __name__ == '__main__':
    main()

