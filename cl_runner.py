from lib_cl import *
import os
is_staging = bool(os.environ.get("STAGING",None))

lg.setLevel(logging.INFO)



cash_for_coins = ['cash for', 'your items', 'will pay', 'buy your', 'pay fair', 
               'your unwanted', 'your jewelry', 'buying sports',
               'years experience', 'selling your', 'cash paid', 'top dollar', 'also buy', 
               'in any condition', 'i pay cash', 'pay top', 'looking to purchase']
piggy_bank_grams = ['piggy bank', 'coin bank', 'ceramic piggy', 'button at', 'piggy banks']
wallet_grams = ['coin purse', 'plenty of space', 'money and important', 
                'slip pockets', 'bifold wallet', 'genuine leather','coin compartment','coin pocket']
crypto_mining = ['the rig', 'power consumption', 'mining rig', 'bit coin']
vending_grams = ['pressed steel', 'used and refurbished', 'are for quarters','machine is', 
                 'machine will', 'soda vending', 'vending machine', 'commercial coin']
coin_sorter = ['coin sorter', 'operating modes','power source', 'fast sort','for each denomination', 'money jar']
arcade = ['coins or freeplay', 'arcade game','arcade games', 'slot machine', 'insert coin', 'arcade cabinet',
          'work on', 'the machine', 'cherry coins', 'beautiful cabinet', 'refurbished with',
          'coinoperated','coinop', 'coin op', 'coin operated', 'coin mat', 'coin flooring', 'coin pusher', 'pinball', 'pin ball']

banned_ngrams_raw = cash_for_coins+piggy_bank_grams+wallet_grams+crypto_mining+vending_grams+coin_sorter+arcade
banned_ngrams = [tuple(i.split()) for i in banned_ngrams_raw]


foreign_areas = [cl_sites_rev[i] for i in ['mendocino.craigslist.org',
'yubasutter.craigslist.org',
'sacramento.craigslist.org',
'stockton.craigslist.org',
'modesto.craigslist.org',
'merced.craigslist.org',
'monterey.craigslist.org',
'goldcountry.craigslist.org',]]

ROW_SZ = 40
aq_pairs = [(i,'coins') for i in [1]+foreign_areas] + [(i,'coin') for i in [1]+foreign_areas]

e = Emailer(pw='ysvvbwadjrinsyyb', fromaddr='davidli276062@gmail.com',
            toaddr=("morgandollar12@gmail.com" if is_staging else "doverdt@gmail.com"))
lp = LexProcessor(rowsize=ROW_SZ)
pdb = PostDb()
imdb = ImDb()
ldb = LexDb(row_size=ROW_SZ)
dbs = [pdb,imdb,ldb]

fp = open("entries.txt", "a")

lg.info(f"Startup (Staging={is_staging})")
while True:
  lg.debug("Start Main Loop")

  new_posts = {}
  for a,q in aq_pairs:
    new_posts = {**new_posts, **fetch_post_data(a,q)}

  for url, pp in new_posts.items():
    for i in dbs:
      i.mgmt()

    pp:ClPosting

    if pdb.query(url) is not None:
      # skip
      lg.debug(f"URL MATCH {url}: {pp.title}")
      continue
    pic_hashes = process_imid_list(pp.im_url_list)
    imres = imdb.query(pic_hashes)
    imdb.add_row(pic_hashes,url)
    lex_df = lp.process(url, clpost=pp)
    pdb.add_row(pp, url)
    if len(pic_hashes) and len(imres)/len(pic_hashes) > 0.5:
      # skip
      lg.info(f"PIC MATCH {url} {pp.title}")
      try:
        lg.info(f"\t- {pdb.df.loc[imres.index[0]].url}")
      except Exception:
        lg.warning("Matching URL not found!")

      if lex_df is not None:
        ldb.add_row(lex_df,url)
      continue
    if lex_df is None:
      lexres = []
    else:
      lexres = ldb.query(lex_df)
    if len(lexres) > 0:
      # skip
      lg.info(f"LEX MATCH {url} {pp.title}")
      try:
        lg.info(f"\t- {pdb.df.loc[lexres.index[0]].url}")
      except Exception:
        lg.warning("Matching URL not found!")

    else:
      ngram_match_list = []
      for j in get_words(pp.innertext) + \
               get_bigrams(pp.innertext) + \
               get_trigrams(pp.innertext) + \
               get_words(pp.title) + \
               get_bigrams(pp.title) + \
               get_bigrams(pp.title):
        if j in banned_ngrams:
          ngram_match_list.append(j)
      
      special_seller = False
      if pp.loc is not None:
        if ((pp.loc[0] - 37.279701)**2 + (pp.loc[1]+121.824203)**2) < 0.005 and pp.price in range(9,40):
          special_seller = True

      if ngram_match_list or special_seller:
        lg.info(f"BANNED {url} {pp.title}")
        lg.info(f"\t- {ngram_match_list}")
      else:
        #print(repr(pp))
        lg.info(f"VALID! {url} {pp.title}")
        fp.write(f"{url}\t{pp.title}\n")
        fp.flush()
        if (time.time() - pp.ts) < (24*3600):
          e.send_email(pp.get_html(url),pp.get_ext_title())
    if lex_df is not None:
      ldb.add_row(lex_df,url)
