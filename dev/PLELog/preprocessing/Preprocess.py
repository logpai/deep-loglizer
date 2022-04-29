from CONSTANTS import *
from entities.instances import Instance
from preprocessing.dataloader.BGLLoader import BGLLoader
from preprocessing.dataloader.HDFSLoader import HDFSLoader


class Preprocessor:
    def __init__(self):
        self.dataloader = None
        self.train_event2idx = {}
        self.test_event2idx = {}
        self.id2label = {}
        self.label2id = {}
        self.templates = []
        self.embedding = None
        self.base = None
        self.dataset = None
        self.parsing = None
        self.tag2id = {'Normal': 0, "Anomaly": 1}
        self.id2tag = {0: 'Normal', 1: "Anomaly"}
        self.logger = self._set_logger()
        pass

    def _set_logger(self):
        # Dispose Loggers.
        PreprocessorLogger = logging.getLogger('Preprocessor')
        PreprocessorLogger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

        file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'Preprocessor.log'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

        PreprocessorLogger.addHandler(console_handler)
        PreprocessorLogger.addHandler(file_handler)
        PreprocessorLogger.info(
            'Construct PreprocessorLogger success, current working directory: %s, logs will be written in %s' %
            (os.getcwd(), LOG_ROOT))
        return PreprocessorLogger

    def process(self, dataset, parsing, template_encoding, cut_func):
        '''
        Preprocess approach, log loading, parsing and cutting.
        Please be noted that if you want to add more datasets or parsers, you should modify here.
        :param dataset: Specified dataset
        :param parsing: Specified log parser, IBM(Drain) now supported.
        :param template_encoding: Semantic representation functio for log templates.
        :param cut_func: Curtting function for all instances.
        :return: Train, Dev and Test data in list of instances.
        '''

        self.base = os.path.join(PROJECT_ROOT, 'datasets/' + dataset + '/inputs/' + parsing)
        self.dataset = dataset
        self.parsing = parsing
        dataloader = None
        parser_config = None
        parser_persistence = os.path.join(PROJECT_ROOT, 'datasets/' + dataset + '/persistences')

        if dataset == 'HDFS':
            dataloader = HDFSLoader(in_file=os.path.join(PROJECT_ROOT, 'datasets/HDFS/HDFS.log'),
                                    semantic_repr_func=template_encoding)
            parser_config = os.path.join(PROJECT_ROOT, 'conf/HDFS.ini')
        elif dataset == 'BGL' or dataset == 'BGLSample':
            in_file = os.path.join(PROJECT_ROOT, 'datasets/' + dataset + '/' + dataset + '.log')
            dataset_base = os.path.join(PROJECT_ROOT, 'datasets/' + dataset)
            dataloader = BGLLoader(in_file=in_file, dataset_base=dataset_base,
                                   semantic_repr_func=template_encoding)
            parser_config = os.path.join(PROJECT_ROOT, 'conf/BGL.ini')

        self.dataloader = dataloader

        if parsing == 'IBM':
            self.dataloader.parse_by_IBM(config_file=parser_config, persistence_folder=parser_persistence)
        else:
            self.logger.error('Parsing method %s not implemented yet.')
            raise NotImplementedError
        return self._gen_instances(cut_func=cut_func)


    def _gen_instances(self, cut_func=None):
        self.logger.info('Start preprocessing dataset %s by parsing method %s' % (self.dataset, self.parsing))
        instances = []
        if not os.path.exists(self.base):
            os.makedirs(self.base)
        train_file = os.path.join(self.base, 'train')
        dev_file = os.path.join(self.base, 'dev')
        test_file = os.path.join(self.base, 'test')

        self.logger.info('Start generating instances.')
        # Prepare semantic embedding sequences for instances.
        for block in tqdm(self.dataloader.blocks):
            if block in self.dataloader.block2eventseq.keys() and block in self.dataloader.block2label.keys():
                id = block
                label = self.dataloader.block2label[id]
                inst = Instance(id, self.dataloader.block2eventseq[id], label)
                instances.append(inst)
            else:
                self.logger.error('Found mismatch block: %s. Please check.' % block)
        self.embedding = self.dataloader.id2embed
        train, dev, test = cut_func(instances)
        self.label_distribution(train, dev, test)
        self.record_files(train, train_file, dev, dev_file, test, test_file)
        self.update_dicts()
        self.update_event2idx_mapping(train, test)
        del self.dataloader
        gc.collect()
        return train, dev, test

    def update_dicts(self):
        self.id2label = self.dataloader.id2label
        self.label2id = self.dataloader.label2id
        self.templates = self.dataloader.templates

    def record_files(self, train, train_file, dev, dev_file, test, test_file, pretrain_source=None):
        with open(train_file, 'w', encoding='utf-8') as writer:
            for instance in train:
                writer.write(str(instance) + '\n')
        if dev:
            with open(dev_file, 'w', encoding='utf-8') as writer:
                for instance in dev:
                    writer.write(str(instance) + '\n')
        with open(test_file, 'w', encoding='utf-8') as writer:
            for instance in test:
                writer.write(str(instance) + '\n')
        if pretrain_source:
            with open(pretrain_source, 'w', encoding='utf-8') as writer:
                for inst in train:
                    writer.write(' '.join([str(x) for x in inst.sequence]) + '\n')

    def label_distribution(self, train, dev, test):
        train_label_counter = Counter([inst.label for inst in train])
        if dev:
            dev_label_counter = Counter([inst.label for inst in dev])
            self.logger.info('Dev: %d Normal, %d Anomalous instances.', dev_label_counter['Normal'],
                             dev_label_counter["Anomaly"])
        test_label_counter = Counter([inst.label for inst in test])
        self.logger.info('Train: %d Normal, %d Anomalous instances.', train_label_counter['Normal'],
                         train_label_counter["Anomaly"])
        self.logger.info('Test: %d Normal, %d Anomalous instances.', test_label_counter['Normal'],
                         test_label_counter["Anomaly"])

    def update_event2idx_mapping(self, pre, post):
        '''
        Calculate unique events in pre & post for event count vector calculation.
        :param pre: pre data, including training set and validation set(if has)
        :param post: post data, mostly testing set
        :return: update mappings in self
        '''
        self.logger.info('Update train instances\' event-idx mapping.')
        pre_ordered_events = self._count_events(pre)
        embed_size = len(pre_ordered_events)
        self.logger.info('Embed size: %d in pre dataset.' % embed_size)
        for idx, event in enumerate(pre_ordered_events):
            self.train_event2idx[event] = idx
        self.logger.info('Update test instances\' event-idx mapping.')
        post_ordered_events = self._count_events(post)
        base = len(pre_ordered_events)
        increment = 0
        for event in post_ordered_events:
            if event not in pre_ordered_events:
                pre_ordered_events.append(event)
                self.test_event2idx[event] = base + increment
                increment += 1
            else:
                self.test_event2idx[event] = self.train_event2idx[event]
        embed_size = len(pre_ordered_events)
        self.logger.info('Embed size: %d in pre+post dataset.' % embed_size)
        pass

    def _count_events(self, sequence):
        events = set()
        for inst in sequence:
            for event in inst.sequence:
                events.add(int(event))
        ordered_events = sorted(list(events))
        return ordered_events
