import hashlib
from collections import Counter


class Instance:
    def __init__(self, block_id, log_sequence, label):
        self.id = block_id
        self.sequence = log_sequence
        self.label = label
        self.repr = None
        self.predicted = ''
        self.confidence = 0
        self.semantic_emb_seq = []
        self.context_emb_seq = []
        self.semantic_emb = None
        self.encode = None
        self.semantic_repr = []
        self.context_repr = []

    def __str__(self):
        sequence_str = ' '.join([str(x) for x in self.sequence])
        if self.predicted == '':
            return sequence_str + '\n' \
                   + str(self.id) + ',' + self.label + '\n'
        else:
            return sequence_str + '\n' \
                   + str(self.id) + ',' + self.label + ',' + self.predicted + ',' + str(self.confidence) + '\n'
        pass

    def __hash__(self):
        return hashlib.md5(str(self).encode('utf-8')).hexdigest()

    @property
    def seq_hash(self):
        return hash(' '.join([str(x) for x in self.sequence]))

    @property
    def event_count(self):
        return Counter(self.sequence)


class Log_With_Datetime:
    def __init__(self, idx, label, datetime, message):
        self.id = idx
        self.label = label
        self.time = datetime
        self.message = message


class Log_Time_Step:
    def __init__(self, logs):
        self.logs = logs
        self.sequence = [log.id for log in self.logs]
        self.label = 'Normal'
        for log in self.logs:
            if log.label == "Anomaly":
                self.label = "Anomaly"
                break
