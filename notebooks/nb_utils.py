import os
import re
import numpy as np
import gensim

CACHE_DIR = os.path.expanduser('~/.cache/dl-cookbook')

def download(url):
    filename = os.path.join(CACHE_DIR, re.sub('[^a-zA-Z0-9.]+', '_', url))
    if os.path.exists(filename):
        return filename
    else:
        os.system('mkdir -p "%s"' % CACHE_DIR)
        assert os.system('wget -O "%s" "%s"' % (filename, url)) == 0
        return filename
    
    
def load_w2v(tokenizer=None):
    word2vec_gz = download('https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz')
    word2vec_vectors = word2vec_gz.replace('.gz', '')
    if not os.path.exists(word2vec_vectors):
        assert os.system('gunzip -d --keep "%s"' % word2vec_gz) == 0
        
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_vectors, binary=True)
    
    total_count = sum(tokenizer.word_counts.values())
    idf_dict = { k: np.log(total_count/v) for (k,v) in tokenizer.word_counts.items() }
    
    w2v = np.zeros((tokenizer.num_words, w2v_model.syn0.shape[1]))
    idf = np.zeros((tokenizer.num_words, 1))

    for k, v in tokenizer.word_index.items():
        if v >= tokenizer.num_words:
            continue

        if k in w2v_model:
            w2v[v] = w2v_model[k]
            idf[v] = idf_dict[k]

    del w2v_model
    return w2v, idf