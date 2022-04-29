import sys

sys.path.extend([".", ".."])
from CONSTANTS import *

PROJECT_ROOT = GET_PROJECT_ROOT()
LOG_ROOT = GET_LOGS_ROOT()
# Dispose Loggers.
VocabLogger = logging.getLogger('Vocab')
VocabLogger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'VocabLogger.log'))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

VocabLogger.addHandler(console_handler)
VocabLogger.addHandler(file_handler)
VocabLogger.info(
    'Construct VocabLogger success, current working directory: %s, logs will be written in %s' %
    (os.getcwd(), LOG_ROOT))


class Vocab(object):
    ##please always set PAD to zero, otherwise will cause a bug in pad filling (Tensor)
    PAD, START, END, UNK = 0, 1, 2, 3

    def __init__(self):
        self._id2tag = []
        self._id2tag.append('Normal')
        self._id2tag.append("Anomaly")
        reverse = lambda x: dict(zip(x, range(len(x))))
        self._tag2id = reverse(self._id2tag)
        if len(self._tag2id) != len(self._id2tag):
            VocabLogger.info("serious bug: output tags dumplicated, please check!")
        VocabLogger.info("Vocab info: #output tags %d" % (self.tag_size))
        self._embed_dim = 0
        self.embeddings=None

    def load_from_dict(self, id2embed):
        '''
        Load word embeddings from the results of preprocessor.
        :param id2embed:
        :return:
        '''
        self._id2word = []
        all_words = set()
        for special_word in ['<pad>', '<bos>', '<eos>', '<oov>']:
            if special_word not in all_words:
                all_words.add(special_word)
                self._id2word.append(special_word)
        for word, embed in id2embed.items():
            self._embed_dim = embed.shape[0]
            all_words.add(word)
            self._id2word.append(word)

        word_num = len(self._id2word)
        VocabLogger.info('Total words: ' + str(word_num) + '\n')
        VocabLogger.info('The dim of pretrained embeddings: %d \n' % (self._embed_dim))
        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)

        oov_id = self._word2id.get('<oov>')
        if self.UNK != oov_id:
            VocabLogger.info("serious bug: oov word id is not correct, please check!")

        embeddings = np.zeros((word_num, self._embed_dim))
        tem_count = 0
        for word, embed in id2embed.items():
            index = self._word2id.get(word)
            vector = np.array(embed, dtype=np.float64)
            embeddings[index] = vector
            embeddings[self.UNK] += vector
            tem_count += 1
        if tem_count != word_num - 4:
            VocabLogger.info("Goes wrong when calculating UNK emb!")
        embeddings[self.UNK] = embeddings[self.UNK] / word_num
        self.embeddings = embeddings

    def load_pretrained_embs(self, embfile):
        embedding_dim = -1
        self._id2word = []
        allwords = set()
        for special_word in ['<pad>', '<bos>', '<eos>', '<oov>']:
            if special_word not in allwords:
                allwords.add(special_word)
                self._id2word.append(special_word)

        with open(embfile, encoding='utf-8') as f:
            line = f.readline()
            vocabSize, embedding_dim = line.strip().split()
            embedding_dim = int(embedding_dim)
            for line in f.readlines():
                values = line.strip().split()
                if len(values) == embedding_dim + 1:
                    curword = values[0]
                    if curword not in allwords:
                        allwords.add(curword)
                        self._id2word.append(curword)
        word_num = len(self._id2word)
        VocabLogger.info('Total words: ' + str(word_num) + '\n')
        VocabLogger.info('The dim of pretrained embeddings: %d \n' % (embedding_dim))

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)

        if len(self._word2id) != len(self._id2word):
            VocabLogger.info("serious bug: words dumplicated, please check!")

        oov_id = self._word2id.get('<oov>')
        if self.UNK != oov_id:
            VocabLogger.info("serious bug: oov word id is not correct, please check!")

        embeddings = np.zeros((word_num, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            # line = f.readline()
            tem_count = 0
            for line in f.readlines():
                values = line.split()
                if len(values) == embedding_dim + 1:
                    index = self._word2id.get(values[0])
                    vector = np.array(values[1:], dtype='float64')
                    embeddings[index] = vector
                    embeddings[self.UNK] += vector
                    tem_count += 1
        if tem_count != word_num - 4:
            VocabLogger.info("Goes wrong when calculating UNK emb!")
        embeddings[self.UNK] = embeddings[self.UNK] / word_num

    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]

    def tag2id(self, xs):
        if isinstance(xs, list):
            return [self._tag2id.get(x) for x in xs]
        return self._tag2id.get(xs)

    def id2tag(self, xs):
        if isinstance(xs, list):
            return [self._id2tag[x] for x in xs]
        return self._id2tag[xs]

    @property
    def vocab_size(self):
        return len(self._id2word)

    @property
    def tag_size(self):
        return len(self._id2tag)

    @property
    def word_dim(self):
        return self._embed_dim