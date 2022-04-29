import sys

sys.path.extend([".", ".."])
from preprocessing.BasicLoader import BasicDataLoader
import os.path

from parsers.Drain_IBM import *


class HDFSLoader(BasicDataLoader):
    def __init__(self, in_file=None, datasets_base=os.path.join(PROJECT_ROOT, 'datasets/HDFS'),
                 semantic_repr_func=None):
        super(HDFSLoader, self).__init__()
        self.id2label = {0: 'Normal', 1: 'Anomaly'}
        self.label2id = {'Normal': 0, 'Anomaly': 1}
        # Dispose Loggers.
        self.logger = logging.getLogger('HDFSLoader')
        self.logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

        file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'HDFSLoader.log'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.info(
            'Construct self.logger success, current working directory: %s, logs will be written in %s' %
            (os.getcwd(), LOG_ROOT))

        self.blk_rex = re.compile(r'blk_[-]{0,1}[0-9]+')
        if not os.path.exists(in_file):
            self.logger.error('Input file not found, please check.')
            exit(1)
        self.in_file = in_file
        self.remove_cols = [0, 1, 2, 3, 4]
        self.dataset_base = datasets_base
        self._load_raw_log_seqs()
        self._load_hdfs_labels()
        self.semantic_repr_func = semantic_repr_func

        # self.label2id = {}
        # self.label2id = {'Normal': 0, 'Anomaly': 1}


    def parse_by_Official(self):
        self._restore()
        templates = [
            "Adding an already existing block (.*)",
            "(.*)Verification succeeded for (.*)",
            "(.*) Served block (.*) to (.*)",
            "(.*):Got exception while serving (.*) to (.*):(.*)",
            "Receiving block (.*) src: (.*) dest: (.*)",
            "Received block (.*) src: (.*) dest: (.*) of size ([-]?[0-9]+)",
            "writeBlock (.*) received exception (.*)",
            "PacketResponder ([-]?[0-9]+) for block (.*) Interrupted\.",
            "Received block (.*) of size ([-]?[0-9]+) from (.*)",
            "PacketResponder (.*) ([-]?[0-9]+) Exception (.*)",
            "PacketResponder ([-]?[0-9]+) for block (.*) terminating",
            "(.*):Exception writing block (.*) to mirror (.*)(.*)",
            "Receiving empty packet for block (.*)",
            "Exception in receiveBlock for block (.*) (.*)",
            "Changing block file offset of block (.*) from ([-]?[0-9]+) to ([-]?[0-9]+) meta file offset to ([-]?[0-9]+)",
            "(.*):Transmitted block (.*) to (.*)",
            "(.*):Failed to transfer (.*) to (.*) got (.*)",
            "(.*) Starting thread to transfer block (.*) to (.*)",
            "Reopen Block (.*)",
            "Unexpected error trying to delete block (.*)\. BlockInfo not found in volumeMap\.",
            "Deleting block (.*) file (.*)",
            "BLOCK\* NameSystem\.allocateBlock: (.*)\. (.*)",
            "BLOCK\* NameSystem\.delete: (.*) is added to invalidSet of (.*)",
            "BLOCK\* Removing block (.*) from neededReplications as it does not belong to any file\.",
            "BLOCK\* ask (.*) to replicate (.*) to (.*)",
            "BLOCK\* NameSystem\.addStoredBlock: blockMap updated: (.*) is added to (.*) size ([-]?[0-9]+)",
            "BLOCK\* NameSystem\.addStoredBlock: Redundant addStoredBlock request received for (.*) on (.*) size ([-]?[0-9]+)",
            "BLOCK\* NameSystem\.addStoredBlock: addStoredBlock request received for (.*) on (.*) size ([-]?[0-9]+) But it does not belong to any file\.",
            "PendingReplicationMonitor timed out block (.*)"
        ]
        save_path = os.path.join(PROJECT_ROOT,
                                 'datasets/HDFS/persistences/official')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        templates_file = os.path.join(save_path, 'NC_templates.txt')
        log2temp_file = os.path.join(save_path, 'log2temp.txt')
        logseq_file = os.path.join(save_path, 'event_seqs.txt')
        if os.path.exists(templates_file) and os.path.exists(log2temp_file) and os.path.exists(logseq_file):
            self.logger.info('Found parsing result, please note that this does not guarantee a smooth execution.')
            with open(templates_file, 'r', encoding='utf-8') as reader:
                self._load_templates(reader)

            with open(log2temp_file, 'r', encoding='utf-8') as reader:
                self._load_log2temp(reader)

            with open(logseq_file, 'r', encoding='utf-8') as reader:
                self._load_log_event_seqs(reader)
            pass
        else:
            self.logger.info('Parsing result not found, start a new one.')
            for id, template in enumerate(templates):
                self.templates[id] = template
            with open(self.in_file, 'r', encoding='utf-8') as reader:
                log_id = 0
                for line in tqdm(reader.readlines()):
                    line = line.strip()
                    if self.remove_cols:
                        processed_line = self._pre_process(line)
                    for index, template in self.templates.items():
                        if re.compile(template).match(processed_line) is not None:
                            self.log2temp[log_id] = index
                            break
                    if log_id not in self.log2temp.keys():
                        self.logger.warning(
                            'Mismatched log message : %s, try using original line.' % processed_line)
                        for index, template in self.templates.items():
                            if re.compile(template).match(line) is not None:
                                self.log2temp[log_id] = index
                                break
                        if log_id not in self.log2temp.keys():
                            self.logger.error('Failed to parse line %s' % line)
                            exit(2)
                    log_id += 1

            for block, seq in self.block2seqs.items():
                self.block2eventseq[block] = []
                for log_id in seq:
                    self.block2eventseq[block].append(self.log2temp[log_id])

            with open(templates_file, 'w', encoding='utf-8') as writer:
                for id, template in self.templates.items():
                    writer.write(','.join([str(id), template]) + '\n')
            with open(log2temp_file, 'w', encoding='utf-8') as writer:
                for logid, tempid in self.log2temp.items():
                    writer.write(','.join([str(logid), str(tempid)]) + '\n')
            with open(logseq_file, 'w', encoding='utf-8') as writer:
                self._save_log_event_seqs(writer)
        self._prepare_semantic_embed(os.path.join(save_path, 'event2semantic.vec'))

    def logger(self):
        return self.logger

    def _pre_process(self, line):
        tokens = line.strip().split()
        after_process = []
        for idx, token in enumerate(tokens):
            if idx not in self.remove_cols:
                after_process.append(token)
        return ' '.join(after_process)

    def _load_raw_log_seqs(self):
        '''
        Load log sequences from raw HDFS log file.
        :return: Update related attributes in current instance.
        '''
        sequence_file = os.path.join(self.dataset_base, 'raw_log_seqs.txt')
        if not os.path.exists(sequence_file):
            self.logger.info('Start extract log sequences from HDFS raw log file.')
            with open(self.in_file, 'r', encoding='utf-8') as reader:
                log_id = 0
                for line in tqdm(reader.readlines()):
                    processed_line = self._pre_process(line)
                    block_ids = set(re.findall(self.blk_rex, processed_line))
                    if len(block_ids) == 0:
                        self.logger.warning('Failed to parse line: %s . Try with raw log message.' % line)
                        block_ids = set(re.findall(self.blk_rex, line))
                        if len(block_ids) == 0:
                            self.logger.error('Failed, please check the raw log file.')
                        else:
                            self.logger.info('Succeed. %d block ids are found.' % len(block_ids))

                    for block_id in block_ids:
                        if block_id not in self.block2seqs.keys():
                            self.blocks.append(block_id)
                            self.block2seqs[block_id] = []
                        self.block2seqs[block_id].append(log_id)

                    log_id += 1
            with open(sequence_file, 'w', encoding='utf-8') as writer:
                for block in self.blocks:
                    writer.write(':'.join([block, ' '.join([str(x) for x in self.block2seqs[block]])]) + '\n')
        else:
            self.logger.info('Start load from previous extraction. File path %s' % sequence_file)
            with open(sequence_file, 'r', encoding='utf-8') as reader:
                for line in tqdm(reader.readlines()):
                    tokens = line.strip().split(':')
                    block = tokens[0]
                    seq = tokens[1].split()
                    if block not in self.block2seqs.keys():
                        self.block2seqs[block] = []
                        self.blocks.append(block)
                    self.block2seqs[block] = [int(x) for x in seq]

        self.logger.info('Extraction finished successfully.')

    def _load_hdfs_labels(self):
        with open(os.path.join(PROJECT_ROOT, 'datasets/HDFS/label.txt'), 'r', encoding='utf-8') as reader:
            reader.readline()
            for line in reader.readlines():
                token = line.strip().split(',')
                block = token[0]
                # label = self.id2label[int(token[1])]
                label = token[1]
                # label = self.label2id[token[1]]
                self.block2label[block] = label


if __name__ == '__main__':
    from representations.templates.statistics import Simple_template_TF_IDF

    semantic_encoder = Simple_template_TF_IDF()
    loader = HDFSLoader(in_file=os.path.join(PROJECT_ROOT, 'datasets/temp_HDFS/HDFS.log'),
                        datasets_base=os.path.join(PROJECT_ROOT, 'datasets/temp_HDFS'),
                        semantic_repr_func=semantic_encoder.present)
    loader.parse_by_IBM(config_file=os.path.join(PROJECT_ROOT, 'conf/HDFS.ini'),
                        persistence_folder=os.path.join(PROJECT_ROOT, 'datasets/temp_HDFS/persistences'))

    pass
