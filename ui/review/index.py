import os
from flask import (
    Blueprint, flash, g, request, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort

from flaskr.db import get_gdb

bp = Blueprint('index', __name__)


reverse_order = {
        'asc': 'desc',
        'desc': 'asc',
        }

@bp.route('/')
def index():
    sort = request.args.get('sort')
    order = request.args.get('order', 'asc')
    order = reverse_order[order]
    page = request.args.get('page', 0)
    gdb = get_gdb()
    samples = gdb.read_samples(sort=sort, order=order, page=page)
    new_samples = []
    for sample in samples[0:99]:
        new_sample = {}
        for k, v in sample.items():
            new_sample[k] = v
        filename =sample['filename']
        new_sample['name'] = os.path.split(filename)[1]
        new_sample['url'] = url_for('static', filename=f'audio/{"/".join(filename.split("/")[6:10])}')
        new_samples.append(new_sample)

    return render_template('index.html', samples=new_samples, sort=sort, order=order)

