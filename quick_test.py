import tqdm
import sqlite3
import os

TABLE = 'samples'
#  TABLE = 'originals'
filepathname = '/mnt/fastest/jem/ml/training_data/processed/raw22/train.sqlite'
filepathname = '/mnt/fastest/jem/ml/training_data/processed/raw22/test.sqlite'
filepathname = '/mnt/fastest/jem/ml/training_data/processed/raw22/validate.sqlite'
con = sqlite3.connect(filepathname)

cur = con.cursor()
results = cur.execute(f"SELECT * from {TABLE}").fetchall()
with tqdm.tqdm(total=len(results), desc='Processing:') as pbar:
    # for r in results[0:20]:
    for r in results:
        # /mnt/fastest/jem/ml/training_data/processed/train/cv_sample-000005_o03300ms.wav
        elements = r[1].split('/')
        #  print(elements)
        if elements[7] == 'raw22':
            print(f'{r[0]} already updated.')
            continue
        elements.insert(7, 'raw22')
        p = '/' + os.path.join(*elements)

        sql = f"UPDATE samples set filename='{p}' where id={r[0]}"
        #  print(sql)
        results = cur.execute(sql)
        pbar.update(1)

con.commit()

#  results = cur.execute(f"ALTER TABLE samples ADD melspectrogram BLOB;")
#  print(results)
