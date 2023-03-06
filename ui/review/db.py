import sqlite3

import click
from flask import current_app, g
from flaskr import data_schema


def init_app(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)

def get_gdb():
    if 'gdb' not in g:
        g.gdb = data_schema.GenderDB()

    return g.gdb


def close_db(e=None):
    gdb = g.pop('gdb', None)

    if gdb is not None:
        gdb.close()
