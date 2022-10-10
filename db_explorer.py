#!/usr/bin/python3 -i

from lib_cl import *

ROW_SZ = 40

e = Emailer()
lp = LexProcessor(rowsize=ROW_SZ)
pdb = PostDb()
imdb = ImDb()
ldb = LexDb(row_size=ROW_SZ)
dbs = [pdb,imdb,ldb]
