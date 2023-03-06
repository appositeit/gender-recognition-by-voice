import datetime
import pytz

import sqlalchemy as db
from sqlalchemy.orm import Session
'''Utility functions for storing and using structured data.'''

LABEL2INT = {"male": 1, "female": 0}


# class CommonVoiceCSV:

#     def __init__(self, csv_path):
#         df = pd.read_csv(csv_path)  # pylint: disable=invalid-name
#         filename_lookup = {}
#         for d in df.itertuples():  # pylint: disable=invalid-name
#             _, base = os.path.split(d.filename)
#             name, _ = os.path.splitext(base)
#             if str(d.gender) == 'nan' or d.gender == 'other':
#                 continue
#             filename_lookup[name] = LABEL2INT[d.gender]

#         self.lookup = filename_lookup



class GenderDB:

    def __init__(self, db_path):
        #  self.connection = self.connect(check_same_thread=False)
        self.connection = self.connect(db_path)
        self.metadata = db.MetaData()
        self.metadata.reflect(self.engine)

        if 'tests' not in self.metadata.tables.keys():
            print('Creating tables')
            self.create_tables()
        self.get_tables()

    def get_tables(self):
        self.originals = db.Table('originals', self.metadata, autoload_with=self.engine)
        self.samples = db.Table('samples', self.metadata, autoload_with=self.engine)
        self.test_runs = db.Table('test_runs', self.metadata, autoload_with=self.engine)
        self.tests = db.Table('tests', self.metadata, autoload_with=self.engine)


    def connect(self,db_path):
        '''Initiate connection to the database.'''
        # engine = create_engine("mysql://gender_recognition:@localhost/gr")
        self.engine = db.create_engine(f'sqlite:///{db_path}')
        return self.engine.connect()

    def close(self):
        self.connection.close()

    def create_tables(self):
        '''Creates the database schema.'''

        # originals = db.Table('originals', self.metadata,
        #               db.Column('id', db.Integer(), db.Identity(), primary_key=True),
        #               db.Column('filename', db.String(255), nullable=False, index=True, unique=True),
        #               db.Column('male', db.Boolean(), default=True),
        #               db.Column('frequency', db.Float(), default=0),
        #               db.Column('confidence', db.Float(), default=0),
        #               db.Column('length', db.Float(), default=0),
        #               )

        # samples = db.Table('samples', self.metadata,
        #               db.Column('id', db.Integer(), db.Identity(), primary_key=True),
        #               db.Column('filename', db.String(255), nullable=False, index=True, unique=True),
        #               db.Column('original_id', db.ForeignKey('originals.id')),
        #               db.Column('parent_offset', db.Float(), default=0),
        #               db.Column('valid', db.Boolean(), default=True),
        #               db.Column('male', db.Boolean(), default=True),
        #               db.Column('frequency', db.Float(), default=0),
        #               db.Column('confidence', db.Float(), default=0),
        #               )

        test_runs = db.Table('test_runs', self.metadata,
                      db.Column('id', db.Integer(), db.Identity()),
                      db.Column('model_filename', db.String(255), nullable=False, index=True, unique=True),
                      db.Column('when', db.DateTime, nullable=False, default=datetime.datetime.now().astimezone(pytz.timezone('Australia/NSW'))),
                      )

        tests = db.Table('tests', self.metadata,
                      db.Column('id', db.Integer(), db.Identity()),
                      db.Column('test_run_id', db.Integer, nullable=False),
                      db.Column('sample_id', db.ForeignKey('samples.id')),
                      db.Column('sample_male', db.Float(), default=1.0),
                      db.Column('test_male', db.Float(), default=1.0),
                      )
        self.metadata.create_all(self.engine, checkfirst=True)  # Creates the table


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

    def read_valid_samples(self, sort=None, order='asc'):
        query = db.select([self.samples]).where(self.samples.columns.valid == True)
        if sort:
            sort_string = sort
            query = db.select([self.samples]).order_by(
                    getattr(getattr(self.samples, sort), order)())
        else:
            query = db.select([self.samples])

        return self.connection.execute(query).mappings().all()

    def read_samples(self, sort=None, order='asc'):
        query = db.select([self.samples])
        if sort:
            sort_string = sort
            query = db.select([self.samples]).order_by(
                    getattr(getattr(self.samples, sort), order)())
        else:
            query = db.select([self.samples])

        return self.connection.execute(query).mappings().all()

    def update_sample(self, id, values):
        query = self.samples.update().values(**values).where(self.samples.c.id == id)
        return self.connection.execute(query)

    def reset_tables(self):
        self.connect(self.db_path)
        self.drop_tables()
        self.create_tables()

    def add_test_run(self, data):
        try:
            query = db.insert(self.test_runs)
            result = self.connection.execute(query, data)
        except db.exc.IntegrityError:
            return None
        return result.inserted_primary_key[0]

    def add_tests(self, data):
        '''Add an array of test entries to the tests table.

        Arguments:
            data: A list of test dicts to add to tests table.
        '''
        try:
            query = db.insert(self.tests)
            _ = self.connection.execute(query, data)
        except db.exc.IntegrityError:
            return None

    def read_tests(self, sort=None, order='asc'):
        query = db.select([self.tests])
        if sort:
            sort_string = sort
            query = db.select([self.tests]).order_by(
                    getattr(getattr(self.tests, sort), order)())
        else:
            query = db.select([self.tests])

        return self.connection.execute(query).mappings().all()

def main():
    # path = '../training_data/processed/raw22/validate.sqlite'
    # gdb = GenderDB(path)
    pass


if __name__ == '__main__':
    main()

