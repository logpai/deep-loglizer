import re
from tqdm import tqdm
from CONSTANTS import *
from utils.common import like_camel_to_tokens

total_words = 0
num_oov = 0

PROJECT_ROOT = GET_PROJECT_ROOT()
LOG_ROOT = GET_LOGS_ROOT()
# Dispose Loggers.
Statistics_Template_Logger = logging.getLogger('Statistics_Template_Encoder')
Statistics_Template_Logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'Statistics_Template.log'))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

Statistics_Template_Logger.addHandler(console_handler)
Statistics_Template_Logger.addHandler(file_handler)
Statistics_Template_Logger.info(
    'Construct Statistics Template Encoder success, current working directory: %s, logs will be written in %s' %
    (os.getcwd(), LOG_ROOT))

class Simple_template_TF_IDF():
    def __init__(self):
        self._word2vec = {}
        self.vocab_size = 0
        self.load_word2vec()

    def not_empty(self, s):
        return s and s.strip()

    def transform(self, words):
        global total_words, num_oov
        if isinstance(words, list):
            return_list = []
            for word in words:
                total_words += 1
                word = word.lower()
                # word = re.sub('[·’!"$%&\'()＃！（）*+,-./:;<=>?，：￥★、…．＞【】［］《》？“”‘\[\\]^_`{|}~]+', '', word)
                if word in self._word2vec.keys():
                    return_list.append(self._word2vec[word])
                else:
                    print(word, end=' ')
                    num_oov += 1
            return np.asarray(return_list, dtype=np.float).sum(axis=0) / len(words)
        else:
            total_words += 1
            word = words.lower()
            # word = re.sub('[·’!"$%&\'()＃！（）*+,-./:;<=>?，：￥★、…．＞【】［］《》？“”‘\[\\]^_`{|}~]+', '', word)
            if word in self._word2vec.keys():
                return self._word2vec[word]
            else:
                num_oov += 1
                return np.zeros(self.vocab_size)

    def load_word2vec(self):
        Statistics_Template_Logger.info('Loading word2vec dict.')
        embed_file = os.path.join(PROJECT_ROOT, 'datasets/glove.6B.300d.txt')
        if os.path.exists(embed_file):
            with open(embed_file, 'r', encoding='utf-8') as reader:
                for line in tqdm(reader.readlines()):
                    tokens = line.strip().split()
                    word = tokens[0]
                    embed = np.asarray(tokens[1:], dtype=np.float)
                    self._word2vec[word] = embed

                    self.vocab_size = len(tokens) - 1

                    if len(tokens) != 301:
                        Statistics_Template_Logger.info('Wow: ' + line)

        else:
            Statistics_Template_Logger.error('No pre-trained embedding file(%s) found. Please check.' % embed_file)
            exit(2)

    def present(self, id2templates):
        processed_id2templates = {}
        all_tokens = set()
        tokens_template_counter = Counter()

        # Preprocessing templates and calculate token-in-template apperance.
        id2embed = {}
        for id, template in id2templates.items():
            # Preprocess: split by spaces and special characters.
            template_tokens = re.split(r'[,\!:=\[\]\(\)\$\s\.\/\#\|\\ ]', template)
            filtered_tokens = []
            for simplified_token in template_tokens:
                if re.match('[\_]+', simplified_token) is not None:
                    filtered_tokens.append('')
                elif re.match('[\-]+', simplified_token) is not None:
                    filtered_tokens.append('')
                else:
                    filtered_tokens.append(simplified_token)
            template_tokens = list(filter(self.not_empty, filtered_tokens))

            # Update token-in-template counter for idf calculation.
            for token in template_tokens:
                tokens_template_counter[token] += 1
                all_tokens = all_tokens.union(template_tokens)

            # Update new processed templates
            processed_id2templates[id] = ' '.join(template_tokens)

        Statistics_Template_Logger.info(
            'Found %d tokens in %d log templates' % (len(all_tokens), len(processed_id2templates)))

        # Calculate IDF score.
        total_templates = len(processed_id2templates)
        token2idf = {}
        for token, count in tokens_template_counter.most_common():
            token2idf[token] = np.log(total_templates / count)

        # Calculate TF score and summarize template embedding.
        for id, template in processed_id2templates.items():
            template_tokens = template.split()
            N = len(template_tokens)
            token_counter = Counter(template_tokens)
            template_emb = np.zeros(self.vocab_size)
            if N == 0:
                id2embed[id] = template_emb
                continue
            for token in template_tokens:
                simple_words = like_camel_to_tokens(token)
                tf = token_counter[token] / N
                if token in token2idf.keys():
                    idf = token2idf[token]
                else:
                    idf = 1
                embed = self.transform(simple_words)
                template_emb += tf * idf * embed
            id2embed[id] = template_emb
        Statistics_Template_Logger.info("OOV Rate: %d/%d = %.4f" % (num_oov, total_words, (num_oov / total_words)))
        return id2embed


class Template_TF_IDF_without_clean():
    def __init__(self):
        self._word2vec = {}
        self.vocab_size = 0
        self.load_word2vec()

    def transform(self, words):
        global num_oov, total_words
        if isinstance(words, list):
            return_list = []
            for word in words:
                total_words += 1
                if word in self._word2vec.keys():
                    return_list.append(self._word2vec[word])
                else:
                    num_oov += 1
                    return_list.append([np.zeros(self.vocab_size)])
            return return_list
        else:
            if words in self._word2vec.keys():
                return self._word2vec[words]
            else:
                return np.zeros(self.vocab_size)

    def load_word2vec(self):
        Statistics_Template_Logger.info('Loading word2vec dict.')

        embed_file = os.path.join(PROJECT_ROOT, 'datasets/glove.6B.300d.txt')
        if os.path.exists(embed_file):
            with open(embed_file, 'r', encoding='utf-8') as reader:
                for line in tqdm(reader.readlines()):
                    tokens = line.strip().split()
                    word = tokens[0]
                    embed = np.asarray(tokens[1:], dtype=np.float)
                    self._word2vec[word] = embed
                    self.vocab_size = len(tokens) - 1
            pass
        else:
            Statistics_Template_Logger.error('No pre-trained embedding file(%s) found. Please check.' % embed_file)
            exit(2)

    def present(self, id2templates):
        templates = []
        ids = []
        all_tokens = set()
        for id, template in id2templates.items():
            templates.append(template)
            ids.append(id)
            all_tokens = all_tokens.union(template.split())
        Statistics_Template_Logger.info('Found %d tokens in %d log templates' % (len(all_tokens), len(templates)))

        # Calculate IDF score.
        total_templates = len(templates)
        assert total_templates == len(ids)
        token2idf = {}
        for token in all_tokens:
            num_include = 0
            for template in templates:
                if token in template:
                    num_include += 1
            idf = math.log(total_templates / (num_include + 1))
            token2idf[token] = idf

        id2embed = {}
        for id, template in id2templates.items():
            template_tokens = template.split()
            N = len(template_tokens)
            token_counter = Counter(template_tokens)
            template_emb = np.zeros(self.vocab_size)
            if N == 0:
                id2embed[id] = template_emb
                continue
            for token in template_tokens:
                tf = token_counter[token] / N
                idf = token2idf[token]
                embed = self.transform(token)
                template_emb += tf * idf * embed
            id2embed[id] = template_emb
        Statistics_Template_Logger.info("Total %d OOV %d" % (total_words, num_oov))
        return id2embed
