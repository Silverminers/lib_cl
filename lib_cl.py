from cl_constants import *

#@title Very General Utils
#@markdown distance, get_url, loggers etc.

import requests
from dataclasses import dataclass
import logging
logging.basicConfig(level = logging.INFO)
lg = logging.getLogger('cl_log')
# lg.setLevel(logging.DEBUG)

def distance(origin, destination):
    if origin is None or destination is None:
      return float("nan")
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d*0.621371 # To miles

# Web fetching

def get_url_text(url):
  for retries in range(URL_MAX_RETRIES):
    try:
      r = requests.get(url, timeout=5)
      if r.status_code == 200:
        val = r.text
      else:
        raise Exception("Bad status code")
      return val
    except Exception as e:
      print(f"[Warn] Failed for: {e}.")
  raise Exception("No retries left!")

URL_MAX_RETRIES = 5

def get_url_raw(url):
  for retries in range(URL_MAX_RETRIES):
    try:
      r = requests.get(url, timeout=5, stream=True)
      if r.status_code == 200:
        val = r.raw.data
      else:
        raise Exception("Bad status code")
      return val
    except Exception as e:
      print(f"[Warn] Failed for: {e}")
      pass
  raise Exception("No retries left!")

def get_url_json(url):
  for retries in range(URL_MAX_RETRIES):
    try:
      r = requests.get(url, timeout=5)
      if r.status_code == 200:
        val = r.json()
      else:
        raise Exception("Bad status code")
      return val
    except Exception as e:
      print(f"[Warn] Failed for: {e}")
      pass
  raise Exception("No retries left!")

@dataclass
class ClPosting:
  title: str
  price: float
  im_url_list: list
  pic_hash_list: list # Not currently used
  loc: tuple
  ts: int
  aliases: list # Not currently used
  innertext: str = None

  def get_ext_title(self):
    ret = self.title
    if self.price != -1:
      ret += f" ${self.price} "
    if self.loc is not None:
      ret = ret + f"({distance(MV, self.loc):.0f}mi)"
    return ret
  
  def get_html(self, the_url):
    clean_innertext = "\n".join(self.innertext.splitlines())
    ret = f'<a href="{the_url}">Link</a><div style="white-space: pre-line">{clean_innertext}</div>'
    for i in self.im_url_list[:2]:
      ret += f'<img src="{i}" title="1" alt="1">'
    return ret

#@title CL parsing
#@markdown fetch_post_data(area_id, query) -> {url => ClPosting}
def parse_location(meta_string):
  try:
    return tuple(map(float,meta_string.split('~')[-2:]))
  except Exception:
    return None
def get_area_tup(loc_list, meta_string):
  ret = loc_list[int(meta_string.split('~')[0].split(":")[0])]
  if len(ret) == 2:
    return (*ret,None)
  else:
    return ret
def get_pic_list(pic_arg):
  try:
    ret = [j[2:] for j in pic_arg[1:] if len(j[2:])>10]
    if not ret:
      return []
    return ret
  except Exception:
    return []
def imid_to_url(imid):
  return f'https://images.craigslist.org/{imid}_300x300.jpg'

def fetch_post_data(area_id, query):
  val = get_url_json(f"https://sapi.craigslist.org/web/v7/postings/search/full?batch={area_id}-0-360-1-0&cc=US&lang=en&query={query}&searchPath=sss&sort=date")

  ret_posts = {}

  if not val['data']['totalResultCount']:
    return ret_posts

  min_post_date = val['data']['decode']['minPostedDate']
  min_post_id = val['data']['decode']['minPostingId']
  loc_list = val['data']['decode']['locations'] # areaId: 1, hostname: 'sfbay', subareaAbbr: 'eby'

  for i in val['data']['items']:
    # print(i)
    title = i[-1]
    price = i[3]
    meta_str = i[4]
    pic_id_list = get_pic_list(i[-2])
    posting_id = i[0]+min_post_id
    timestamp = i[1] + min_post_date
    posting_cat = cat_rev[i[2]]
    area_id, hostname, subarea_abbr = get_area_tup(loc_list, meta_str)

    if subarea_abbr is not None:
      url = f'https://{hostname}.craigslist.org/{subarea_abbr}/{posting_cat}/{posting_id}.html'
    else:
      url = f'https://{hostname}.craigslist.org/{posting_cat}/{posting_id}.html'

    ret_posts[url] = ClPosting(title=title, price=price, loc=parse_location(meta_str),ts=timestamp, 
                                pic_hash_list=[], im_url_list=[imid_to_url(i) for i in pic_id_list], aliases=[])
  return ret_posts

#@title Image processing
#@markdown process_imid_list([im_url]) -> [hash]
import time
from multiprocessing.dummy import Pool as thread_pool
from PIL import Image 
import hashlib
import cv2
import numpy as np
import io 


tp = thread_pool(20)

def equalize(img_in):
  # segregate color streams
  b,g,r = cv2.split(img_in)
  h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
  h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
  h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
  # calculate cdf    
  cdf_b = np.cumsum(h_b)  
  cdf_g = np.cumsum(h_g)
  cdf_r = np.cumsum(h_r)
    
  # mask all pixels with value=0 and replace it with mean of the pixel values 
  cdf_m_b = np.ma.masked_equal(cdf_b,0)
  cdf_m_b = (cdf_m_b - cdf_m_b.min())*255/(cdf_m_b.max()-cdf_m_b.min())
  cdf_final_b = np.ma.filled(cdf_m_b,0).astype('uint8')
  
  cdf_m_g = np.ma.masked_equal(cdf_g,0)
  cdf_m_g = (cdf_m_g - cdf_m_g.min())*255/(cdf_m_g.max()-cdf_m_g.min())
  cdf_final_g = np.ma.filled(cdf_m_g,0).astype('uint8')
  cdf_m_r = np.ma.masked_equal(cdf_r,0)
  cdf_m_r = (cdf_m_r - cdf_m_r.min())*255/(cdf_m_r.max()-cdf_m_r.min())
  cdf_final_r = np.ma.filled(cdf_m_r,0).astype('uint8')
  # merge the images in the three channels
  img_b = cdf_final_b[b]
  img_g = cdf_final_g[g]
  img_r = cdf_final_r[r]
  
  img_out = cv2.merge((img_b, img_g, img_r))
  return img_out

def process(pic):
  DISCRETIZE_LEVEL = 4
  discretize_factor = 256/DISCRETIZE_LEVEL
  resized_pic = cv2.resize(pic.astype(np.uint8), dsize=(25, 25), interpolation=cv2.INTER_CUBIC)
  norm_pic  = equalize(resized_pic.astype(np.uint8)).astype(np.uint8)
  return ((norm_pic//discretize_factor)*discretize_factor).astype(np.uint8)

def get_img(url):
  v = get_url_raw(url)
  return v

def turn_bytes_into_img(r_data):
  v = np.array(Image.open(io.BytesIO(r_data)))
  return v

def hash_im(im):
  hash = hashlib.blake2b(im.tobytes(), digest_size=8)
  return np.uint64(int.from_bytes(hash.digest(), 'little'))

def img_to_hash(im):
  return hash_im(process(im))


def process_imid_list(ll):
  st = time.perf_counter()
  im_raw = tp.map(get_img, ll)
  # print(f"Process Imid: {time.perf_counter()-st}")
  images = map(turn_bytes_into_img, im_raw)
  return list(map(img_to_hash, images))

#@title Lexical processing
#@markdown lex_process_url(url,rowsize=None) -> Df[Index(hash), word, ll, freq, wt] | None \
#@markdown Note that the df is indexed by hash of the word, modulo row size if specified.
from bs4 import BeautifulSoup
import re
import html
import pandas as pd
from collections import defaultdict
import string 
from time import perf_counter

PUNCTUATION_FILTER = re.compile(r"([\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~]{3,})")
DEBUG_LEXER = False

word_freq_dataset_url = 'https://raw.githubusercontent.com/garyongguanjie/entrie/main/unigram_freq.csv'
wordfreq = pd.read_csv(word_freq_dataset_url)
wordfreq['ll']=np.log(wordfreq['count']/wordfreq['count'].sum())
wordfreq.index = wordfreq.word
wf = wordfreq.ll.to_dict()
wf = {k:v for k,v in wf.items() if isinstance(k, str)}

@np.vectorize
def hash_st(st):
  hash = hashlib.blake2b(bytes(st, 'ascii'), digest_size=8)
  return np.uint64(int.from_bytes(hash.digest(), 'little'))

def parse_html_for_text(html_str):
  soup = BeautifulSoup(html_str, features='html.parser')
  sp = soup.find(id='postingbody')
  if sp is None:
    return ""
  else:
    return sp.get_text(separator=' ').replace("QR Code Link to This Post", '')

delchars = ''.join(c for c in map(chr, range(256)) if not c.isalnum())
def get_alphanumeric(s):
  return s.translate(str.maketrans('', '', delchars))
def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)
def has_letters(inputString):
    return any(char.isalpha() for char in inputString)
def get_ll_entropy(s):
  if PUNCTUATION_FILTER.match(s):
    return (-4.21950770518)*len(s) # Log(1/68) , since there are 32 punctuation characters 
  ss = get_alphanumeric(s)
  if has_numbers(ss) and has_letters(ss):
    return (-3.58351893846)*len(ss) # log(1/36)
  elif has_numbers(ss):
    return (-2.30258509299)*len(ss) # log(1/10)
  else:
    return (-3.25809653802)*len(ss) # log(1/26)

def innertext_to_word_soup(innertext):
  escaped_innertext = html.unescape(innertext).encode("ascii", errors="ignore").decode()
  
  punc = PUNCTUATION_FILTER.findall(escaped_innertext)

  # Preprocess step: remove certain punctuation
  splitter_chars = [',','.',';','\\','/','-']
  for i in splitter_chars:
    escaped_innertext = escaped_innertext.replace(i,' ')

  words = [i.lower() for i in escaped_innertext.split()]
  words = words + punc # add the punctuation "Words."

  wd = defaultdict(int)
  for i in words:
    # transformed = (i.translate(str.maketrans('', '', string.punctuation)))
    if PUNCTUATION_FILTER.match(i):
      transformed = i
    else:
      transformed = i.strip(string.punctuation).replace("'","") # Special case of removing apostraphe because word database doesn't use apostraphe
    if transformed:      
      if transformed != i:
        if DEBUG_LEXER:
          print(f'\t{i}->{transformed}')
      wd[transformed]+= 1

  wd_to_ll = {}
  for word,freq in wd.items():
    try:
      # print(word, freq*wf[word])
      wd_to_ll[word] = (wf[word], freq)
    except Exception:
      if DEBUG_LEXER:
        print(f"\t{word} not found")
      wd_to_ll[word] = (get_ll_entropy(word), freq)
  return sorted([(k,*v) for k,v in wd_to_ll.items()], key=lambda x:x[1])

def ws_to_df(ws, row_size=None):
  g = pd.DataFrame(ws, columns=['word','ll','freq'])
  g['wt'] = g.ll*g.freq
  g.sort_values('wt',ascending=True, inplace=True)
  g['wt'] = g.wt#/np.sqrt((g.wt**2).sum())
  g.index = hash_st(g.word)
  if row_size is not None:
    g.index = g.index % row_size
    g = g[~g.index.duplicated(keep='first')].copy()
  return g

class LexProcessor:
  def __init__(self, rowsize=None, rl=1.0):
    self.next_process_time = 0
    self.rowsize = rowsize
    self.rl = rl
  def get_text(self, url):
    while perf_counter() < self.next_process_time:
      pass
    h = get_url_text(url)
    self.next_process_time = perf_counter() + self.rl
    innertext = parse_html_for_text(h)
    return innertext
  def process(self, url, clpost=None):
    while perf_counter() < self.next_process_time:
      pass
    h = get_url_text(url)
    self.next_process_time = perf_counter() + self.rl
    innertext = parse_html_for_text(h)
    if clpost is not None:
      clpost.innertext = innertext
    ws = innertext_to_word_soup(innertext)
    if not ws:
      return None
    return ws_to_df(ws, row_size=self.rowsize)

#@title DB Abc
import os.path
from os import rename
from abc import ABC, abstractmethod
import time
import math

def itime():
  return int(time.time())

def atime():
  return np.uint32(time.time())

def get_next_cy(n, period, bias):
  return int(period*math.ceil((n+1-bias)/period))+bias

class DbMgr(ABC):
  @abstractmethod
  def get_schema(self):
    ''' Dict of col => type'''
    pass

  @abstractmethod
  def add_row(self, rw_repr, url):
    pass

  @abstractmethod
  def query(self, url):
    pass

  def create_db(self):
    self.df = pd.DataFrame({**{colname: pd.Series(dtype=dtype) for colname, dtype in self.get_schema().items()}, **{"ts":pd.Series(dtype=np.uint32)}})
    self.df.index = self.df.index.astype(np.uint64)

  def verify_load(self):
    try:
      for colname,dtype in self.get_schema().items():
        self.df[colname].astype(dtype)
      self.df['ts'].astype(np.uint32)
    except KeyError:
      return False
    return True

  def backup(self):
    if self.fn is None:
      return
    sv_fn = f'{self.fn}_new'
    self.df.to_pickle(sv_fn)
    rename(sv_fn, self.fn)
  
  def gc(self):
    if self.gc_thresh_sec is None:
      return
    time_back = itime() - self.gc_thresh_sec
    self.df.drop(self.df.index[self.df.ts < time_back], inplace=True)
  
  def mgmt(self):
    if time.time() >= self.next_gc_time:
      lg.info(f"[{self.fn} mgmt]: Doing Garbage Collection")
      self.gc()
      self.next_gc_time = get_next_cy(itime(), self.gc_timer_interval,self.gc_timer_mod)
    if time.time() >= self.next_bu_time:
      lg.info(f"[{self.fn} mgmt]: Doing backup")
      self.backup()
      self.next_bu_time = get_next_cy(itime(), self.backup_timer_interval,self.backup_timer_mod)

  @property
  def innerdf(self):
    return self.df[[i for i in self.df.columns if i != 'ts']]

  def __setitem__(self,idx,v):
    self.df.loc[idx] = list(v) + [itime()]
  def __getitem__(self,idx):
    return self.df.loc[idx]

  def __init__(self, fn, gc_thresh_sec=None, gc_timer_interval=600, gc_timer_mod=300, backup_timer_interval=600, backup_timer_mod=0):
    self.fn = fn
    self.gc_thresh_sec = gc_thresh_sec
    self.gc_timer_interval = gc_timer_interval
    self.gc_timer_mod = gc_timer_mod
    self.backup_timer_interval = backup_timer_interval
    self.backup_timer_mod = backup_timer_mod
    t = itime()
    self.next_gc_time = get_next_cy(t, gc_timer_interval,gc_timer_mod)
    self.next_bu_time = get_next_cy(t, backup_timer_interval,backup_timer_mod)
    if fn is not None and os.path.isfile(fn):
      self.df = pd.read_pickle(self.fn)
      if not self.verify_load():
        lg.warning(f"DB Verification failed; starting new DB: {self.fn}!")
        self.create_db()
    else:
      self.create_db()

#@title PostDb

class PostDb(DbMgr):
  def get_schema(self):
    return {'title':'object', 'url':'object', 'price':np.int32, 'lat':np.float32, 'lon':np.float32}
  def add_row(self,clposting, url):
    hurl = np.uint64(hash_st(url))
    self.df.drop(int(hurl), inplace=True,errors='ignore')
    cols = {'title':clposting.title, 'url':url, 'price':np.int32(clposting.price), 'lat':np.nan, 'lon':np.nan, 'ts':atime()}
    if clposting.loc is not None:
      cols['lat'] = np.float32(clposting.loc[0])
      cols['lon'] = np.float32(clposting.loc[1])
    self.df = pd.concat( [self.df, pd.DataFrame(cols, index=np.array([hurl],dtype=np.uint64))] )
  def query(self, url):
    hurl = np.uint64(hash_st(url))
    try:
      ret = self.df.loc[hurl]
    except KeyError:
      return None
    self.df.loc[hurl,'ts'] = atime() # Mark the use.
    return ret
  def __init__(self, fn='posts_url.pkl'):
    return super().__init__(fn, gc_thresh_sec=(90*24*3600), gc_timer_mod=100, backup_timer_mod=200) # Garbage collection threshold of 3 months.

#@title ImDb

class ImDb(DbMgr):
  def get_schema(self):
    return {'post':np.uint64}
  def add_row(self,pic_hash_list, url):
    hurl = np.uint64(hash_st(url))
    self.df = pd.concat([self.df, pd.DataFrame({'post':hurl, 'ts':itime()}, index=np.array(pic_hash_list,dtype=np.uint64))])
    self.df = self.df[~self.df.index.duplicated(keep='last')].copy()
  def query(self,pic_hash_list):
    my_df = pd.DataFrame( index=np.array(pic_hash_list,dtype=np.uint64))
    joined = self.df.join(my_df,how='inner')
    self.df.loc[joined.index,'ts'] = itime() # Mark the use.
    # swap the index and 'post' element.
    joined['imhash'] = joined.index
    joined.index = joined.post
    del joined['post']
    return joined
  def __init__(self, fn='im_db.pkl'):
    return super().__init__(fn, gc_thresh_sec=(90*24*3600), gc_timer_mod=300, backup_timer_mod=400) # Garbage collection threshold of 3 months.

#@title LexDb
class LexDb(DbMgr):
  def parse_post_df(self, df):
    if self.use_compressed_word:
      words = [np.uint64(0) for i in range(self.row_size)]
    else:
      words = ['' for i in range(self.row_size)]
    lls = [np.float32(0) for i in range(self.row_size)]
    freqs = [np.uint16(0) for i in range(self.row_size)]
    for idx,rw in df.iterrows():
      words[idx] = np.uint64(hash_st(rw.word)) if self.use_compressed_word else rw.word
      lls[idx] = np.float32(rw.ll)
      freqs[idx] = np.uint16(rw.freq)
    return words,lls,freqs

  def add_row(self, df, url):
    words,lls,freqs = self.parse_post_df(df)

    cols = {}
    for idx,i in enumerate(words):
      cols[f'w{idx}'] = i
    for idx,i in enumerate(lls):
      cols[f'll{idx}'] = i
    for idx,i in enumerate(freqs):
      cols[f'cnt{idx}'] = i
    if self.include_url:
      cols['url'] = url
    cols['ts'] = itime()

    hurl = np.uint64(hash_st(url))
    self.df.drop(int(hurl), inplace=True,errors='ignore')
    self.df = pd.concat( [self.df, pd.DataFrame(cols, index=np.array([hurl],dtype=np.uint64))] )

  def _query(self, words, lls, freqs, COS_THRESH=0.7, WEIGHT_THRESH=-150, MATCH_WORD_THRESH=4):
    w_match_mask = (self.df.iloc[:,self.WORDS_IDX] == words).to_numpy()

    db_wts = self.df.iloc[:,self.LL_IDX].to_numpy()*self.df.iloc[:,self.FREQ_IDX].to_numpy()
    db_norm_factor = np.sqrt((db_wts**2).sum(axis=1))
    qry_wts = (np.array(lls) * np.array(freqs))
    qry_norm_factor = np.sqrt((qry_wts**2).sum())

    cos_sim = (qry_wts * db_wts * w_match_mask).sum(axis=1)/(db_norm_factor*qry_norm_factor)

    match_wts = (np.minimum(self.df.iloc[:,self.FREQ_IDX].to_numpy(),np.array(freqs)) * w_match_mask * lls).sum(axis=1)
    num_matched_words = w_match_mask.sum(axis=1)

    match_mask = (cos_sim>COS_THRESH)&(match_wts<WEIGHT_THRESH)&(num_matched_words>=MATCH_WORD_THRESH)
    if self.include_url:
      res = self.df.iloc[match_mask].url.to_frame()
    else:
      res = self.df.iloc[match_mask].index.to_frame()
      del res[0]
    self.df.loc[match_mask,'ts'] = itime() # Mark the use.

    res['cos'] = cos_sim[match_mask]
    res['match_weight'] = match_wts[match_mask]
    res['num_match_words'] = num_matched_words[match_mask]
    return res

  def query(self, df, COS_THRESH=0.7, WEIGHT_THRESH=-150, MATCH_WORD_THRESH=4):
    words,lls,freqs = self.parse_post_df(df)
    return self._query(words,lls,freqs, COS_THRESH=COS_THRESH, WEIGHT_THRESH=WEIGHT_THRESH, MATCH_WORD_THRESH=MATCH_WORD_THRESH)

  def query_from_rw(self, rw, COS_THRESH=0.7, WEIGHT_THRESH=-150, MATCH_WORD_THRESH=4):
    assert(rw.dtype == object)
    words = rw.loc['w0':f'w{self.row_size-1}'].to_numpy()
    lls = rw.loc['ll0':f'll{self.row_size-1}'].to_numpy()
    freqs = rw.loc['cnt0':f'cnt{self.row_size-1}'].to_numpy()

    return self._query(words,lls,freqs, COS_THRESH=COS_THRESH, WEIGHT_THRESH=WEIGHT_THRESH, MATCH_WORD_THRESH=MATCH_WORD_THRESH)


  def get_schema(self):
    cols = {}
    if self.use_compressed_word:
      for i in range(self.row_size):
         cols[f'w{i}'] = np.uint64
    else:
      for i in range(self.row_size):
        cols[f'w{i}'] = "object"
    for i in range(self.row_size):
      cols[f'll{i}'] = np.float32
    for i in range(self.row_size):
      cols[f'cnt{i}'] = np.uint16
    if self.include_url:
      cols['url'] = "object"
    return cols

  def __init__(self, fn='lex_db.pkl', use_compressed_word=True, include_url=False, row_size=40):
    self.use_compressed_word = use_compressed_word
    self.include_url = include_url
    self.row_size = row_size
    self.WORDS_IDX = (slice(0,row_size))
    self.LL_IDX = (slice(row_size,2*row_size))
    self.FREQ_IDX = (slice(2*row_size, 3*row_size))
    return super().__init__(fn, gc_thresh_sec=(90*24*3600), gc_timer_mod=500, backup_timer_mod=0) # Garbage collection threshold of 3 months.

#@title ngram processing: text_list_to_ngram_df(list[str], n), get_bigrams, get_trigrams
def sep_ln_into_words(ln):
  ln = re.sub(r"[\!\"\#\$\%\&\\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~]", "", ln)
  splitter_chars = [',','.',';','\\','/','-']
  ln = ln.encode("ascii", errors="ignore").decode()
  for i in splitter_chars:
    ln = ln.replace(i,' ')

  return [i.lower().replace("'","") for i in ln.split()]

def sep_text_into_lines(text):
  return re.split(r"[\!&\(\)\*\+\,\.\/\:\;\<\=\>\?\[\\\]\^\_\{\|\}\~\n]+",text)

def get_bigrams(text):
  grams = [sep_ln_into_words(i) for i in sep_text_into_lines(text) if len(sep_ln_into_words(i))]
  return [b for l in grams for b in zip(l[:-1], l[1:])]

def get_trigrams(text):
  grams = [sep_ln_into_words(i) for i in sep_text_into_lines(text) if len(sep_ln_into_words(i))]
  return [b for l in grams for b in zip(l[0:-2], l[1:-1],l[2:])]

def text_list_to_ngram_df(text_list, n=2):
  bgs = defaultdict(int)
  for ps in text_list:
    grams_list = get_bigrams(ps) if n==2 else get_trigrams(ps)
    for i in grams_list:
      bgs[i] += 1
  return pd.DataFrame({'freq':bgs.values()}, index=bgs.keys())


#@title Email Processor
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import  encoders
import smtplib
from time import perf_counter

PW = 'ogzxjnwrylmvkczv'
FROMADDR = 'davidli27606@gmail.com'
TOADDR = 'doverdt@gmail.com'

class Emailer:
  def __init__(self, pw=PW, fromaddr=FROMADDR, toaddr=TOADDR, rl=5.0):
    self.next_process_time = 0
    self.pw = pw
    self.fromaddr = fromaddr
    self.toaddr = toaddr
    self.rl = rl

  def _send_email(self, content, title, fromaddr=FROMADDR, mdpfrom=PW, toaddr=TOADDR, filename=None, filepath=None):
    msg = MIMEMultipart()  # instance of MIMEMultipart
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = title

    body_email = content
    msg.attach(MIMEText(body_email, 'html'))
    if filename is not None:
      attachment = open(filepath, 'rb')  # open the file to be sent
      p = MIMEBase('application', 'octet-stream')  # instance of MIMEBase
      p.set_payload(attachment.read())
      encoders.encode_base64(p)
      p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
      msg.attach(p)  # attach the instance 'p' to instance 'msg'

    s = smtplib.SMTP('smtp.gmail.com', 587)  # SMTP
    s.starttls()
    s.login(fromaddr, mdpfrom)

    text = msg.as_string()

    s.sendmail(fromaddr, toaddr, text)  # sending the email

    s.quit()  # terminating the session

  def send_email(self, content, title):
    while perf_counter() < self.next_process_time:
      pass
    self._send_email(content, title, fromaddr=self.fromaddr, mdpfrom=self.pw, toaddr=self.toaddr)
    self.next_process_time = perf_counter() + self.rl
