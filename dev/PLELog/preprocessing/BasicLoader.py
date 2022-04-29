import sys

sys.path.extend([".", ".."])
from CONSTANTS import *
from parsers.Drain_IBM import Drain3Parser
import abc


def _async_parsing(parser, lines, log2temp):
    for id, line in lines:
        cluster = parser.match(line)
        log2temp[id] = cluster.cluster_id


class BasicDataLoader():
    def __init__(self):
        self.in_file = None
        self.logger = None
        self.block2emb = {}
        self.blocks = []
        self.templates = {}
        self.log2temp = {}
        self.rex = []
        self.remove_cols = []
        self.id2label = {0: 'Normal', 1: "Anomaly"}
        self.label2id = {'Normal': 0, "Anomaly": 1}
        self.block_set = set()
        self.block2seqs = {}
        self.block2label = {}
        self.block2eventseq = {}
        self.id2embed = {}
        self.semantic_repr_func = None

    @abc.abstractmethod
    def _load_raw_log_seqs(self):
        return

    @abc.abstractmethod
    def logger(self):
        return

    @abc.abstractmethod
    def _pre_process(self, line):
        return

    def parse_by_IBM(self, config_file, persistence_folder, core_jobs=-1):
        '''
        Load parsing results by IDM Drain
        :param config_file: IDM Drain configuration file.
        :param persistence_folder: IDM Drain persistence file.
        :return: Update templates, log2temp attributes in self.
        '''
        self._restore()
        if not os.path.exists(config_file):
            self.logger.error('IBM Drain config file %s not found.' % config_file)
            exit(1)
        parser = Drain3Parser(config_file=config_file, persistence_folder=persistence_folder)
        persistence_folder = parser.persistence_folder

        # Specify persistence files.
        log_event_seq_file = os.path.join(persistence_folder, 'log_sequences.txt')
        log_template_mapping_file = os.path.join(persistence_folder, 'log_event_mapping.dict')
        templates_embedding_file = os.path.join(parser.persistence_folder, 'templates.vec')
        start_time = time.time()
        if parser.to_update:
            self.logger.info('No trained parser found, start training.')
            parser.parse_file(self.in_file, remove_cols=self.remove_cols)
            self.logger.info('Get total %d templates.' % len(parser.parser.drain.clusters))

        # Load templates from trained parser.
        for cluster_inst in parser.parser.drain.clusters:
            self.templates[int(cluster_inst.cluster_id)] = cluster_inst.get_template()

        # check parsing resutls such as log2event dict and template embeddings.
        if self._check_parsing_persistences(log_template_mapping_file, log_event_seq_file):
            self.load_parsing_results(log_template_mapping_file, log_event_seq_file)
            pass
        else:
            # parsing results not found, or somehow missing.
            self.logger.info('Missing persistence file(s), start with a full parsing process.')
            self.logger.warning(
                'If you don\'t want this to happen, please copy persistence files from somewhere else and put it in %s' % persistence_folder)
            ori_lines = []
            with open(self.in_file, 'r', encoding='utf-8') as reader:
                log_id = 0
                for line in tqdm(reader.readlines()):
                    processed_line = self._pre_process(line)
                    ori_lines.append((log_id, processed_line))
                    log_id += 1
            self.logger.info('Parsing raw log....')
            if core_jobs != -1:
                m = Manager()
                log2temp = m.dict()
                pool = Pool(core_jobs)
                splitted_lines = self._split(ori_lines, core_jobs)
                inputs = zip([parser] * core_jobs, splitted_lines, [log2temp] * core_jobs)
                pool.starmap(_async_parsing, inputs)
                pool.close()
                pool.join()
                self.log2temp = dict(log2temp)
                pass
            else:
                for item in ori_lines:
                    self._sync_parsing(parser, item)

            self.logger.info('Finished parsing in %.2f' % (time.time() - start_time))

            # Transform original log sequences with log ids(line number) to log event sequence.
            for block, seq in self.block2seqs.items():
                self.block2eventseq[block] = []
                for log_id in seq:
                    self.block2eventseq[block].append(self.log2temp[log_id])

            # Record block id and log event sequences.
            self._record_parsing_results(log_template_mapping_file, log_event_seq_file)
        # Prepare semantic embeddings.
        self._prepare_semantic_embed(templates_embedding_file)
        self.logger.info('All data preparation finished in %.2f' % (time.time() - start_time))

    def load_parsing_results(self, log_template_mapping_file, event_seq_file):
        self.logger.info('Start loading previous parsing results.')
        start = time.time()
        log_template_mapping_reader = open(log_template_mapping_file, 'r', encoding='utf-8')
        event_seq_reader = open(event_seq_file, 'r', encoding='utf-8')
        self._load_log2temp(log_template_mapping_reader)
        self._load_log_event_seqs(event_seq_reader)
        log_template_mapping_reader.close()
        event_seq_reader.close()
        self.logger.info('Finished in %.2f' % (time.time() - start))

    def _restore(self):
        self.block2emb = {}
        self.templates = {}
        self.log2temp = {}

    def _save_log_event_seqs(self, writer):
        self.logger.info('Start saving log event sequences.')
        for block, event_seq in self.block2eventseq.items():
            event_seq = map(lambda x: str(x), event_seq)
            seq_str = ' '.join(event_seq)
            writer.write(str(block) + ':' + seq_str + '\n')
        self.logger.info('Log event sequences saved.')

    def _load_log_event_seqs(self, reader):
        for line in reader.readlines():
            tokens = line.strip().split(':')
            block = tokens[0]
            seq = tokens[1].split()
            self.block2eventseq[block] = [int(x) for x in seq]
        self.logger.info('Loaded %d blocks' % len(self.block2eventseq))

    def _prepare_semantic_embed(self, semantic_emb_file):
        if self.semantic_repr_func:
            self.id2embed = self.semantic_repr_func(self.templates)
            with open(semantic_emb_file, 'w', encoding='utf-8') as writer:
                for id, embed in self.id2embed.items():
                    writer.write(str(id) + ' ')
                    writer.write(' '.join([str(x) for x in embed.tolist()]) + '\n')
            self.logger.info(
                'Finish calculating semantic representations, please found the vector file at %s' % semantic_emb_file)
        else:
            self.logger.warning(
                'No template encoder. Please be NOTED that this may lead to duplicate full parsing process.')

        pass

    def _check_parsing_persistences(self, log_template_mapping_file, event_seq_file):
        flag = self._check_file_existence_and_contents(
            log_template_mapping_file) and self._check_file_existence_and_contents(event_seq_file)
        return flag

    def _check_file_existence_and_contents(self, file):
        flag = os.path.exists(file) and os.path.getsize(file) != 0
        self.logger.info('checking file %s ... %s' % (file, str(flag)))
        return flag

    def _record_parsing_results(self, log_template_mapping_file, evet_seq_file):
        # Recording IBM parsing result.
        start_time = time.time()
        log_template_mapping_writer = open(log_template_mapping_file, 'w', encoding='utf-8')
        event_seq_writer = open(evet_seq_file, 'w', encoding='utf-8')
        self._save_log2temp(log_template_mapping_writer)
        self._save_log_event_seqs(event_seq_writer)
        log_template_mapping_writer.close()
        event_seq_writer.close()
        self.logger.info('Done in %.2f' % (time.time() - start_time))

    def _load_templates(self, reader):
        for line in reader.readlines():
            tokens = line.strip().split(',')
            id = tokens[0]
            template = ','.join(tokens[1:])
            self.templates[int(id)] = template
        self.logger.info('Loaded %d templates' % len(self.templates))

    def _save_templates(self, writer):
        for id, template in self.templates.items():
            writer.write(','.join([str(id), template]) + '\n')
        self.logger.info('Templates saved.')

    def _load_log2temp(self, reader):
        for line in reader.readlines():
            logid, tempid = line.strip().split(',')
            self.log2temp[int(logid)] = int(tempid)
        self.logger.info('Loaded %d log sequences and their mappings.' % len(self.log2temp))

    def _save_log2temp(self, writer):
        for log_id, temp_id in self.log2temp.items():
            writer.write(str(log_id) + ',' + str(temp_id) + '\n')
        self.logger.info('Log2Temp saved.')

    def _load_semantic_embed(self, reader):
        for line in reader.readlines():
            token = line.split()
            template_id = int(token[0])
            embed = np.asarray(token[1:], dtype=np.float)
            self.id2embed[template_id] = embed
        self.logger.info('Load %d templates with embedding size %d' % (len(self.id2embed), self.id2embed[1].shape[0]))

    def _split(self, X, copies=5):
        quota = int(len(X) / copies) + 1
        res = []
        for i in range(copies):
            res.append(X[i * quota:(i + 1) * quota])
        return res

    def _sync_parsing(self, parser, line):
        (id, message) = line
        cluster = parser.match(message)
        self.log2temp[id] = cluster.cluster_id
