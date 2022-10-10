from lib_cl import *

from flask import Flask

app = Flask(__name__)

# {'title':'object', 'url':'object', 'price':np.int32, 'lat':np.float32, 'lon':np.float32, 
#               'pic1':'object', 'pic2': 'object', 'innertext':'object'}


@app.route('/')
def hello():
    buf = []
    pdb = PostDb()
    for idx,i in pdb.df.iterrows():
        pp = ClPosting.from_post_rw(i)
        buf.append(pp.get_html(i.url))
        buf.append("<hr>")
    return "\n".join(buf)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)