import torch
from torch.autograd import Variable

class TInstWithLogits:
    def __init__(self, batch_size, slen, tag_size):
        self.src_ids = []
        self.src_words = Variable(torch.LongTensor(batch_size, slen).zero_(), requires_grad=False)
        self.src_masks = Variable(torch.Tensor(batch_size, slen).zero_(), requires_grad=False)
        self.tags = Variable(torch.FloatTensor(batch_size, tag_size).zero_(), requires_grad=False)
        self.g_truth = Variable(torch.LongTensor(batch_size).zero_(), requires_grad=False)
        self.word_len = Variable(torch.LongTensor(batch_size).zero_(), requires_grad=False)

    def to_cuda(self, device):
        if str(device) != "cpu":
            self.src_words = self.src_words.cuda(device)
            self.src_masks = self.src_masks.cuda(device)
            self.tags = self.tags.cuda(device)
            self.g_truth = self.g_truth.cuda(device)
            self.word_len = self.word_len.cuda(device)

    @property
    def inputs(self):
        return self.src_words, self.src_masks, self.word_len

    @property
    def ids(self):
        return self.src_ids

    @property
    def targets(self):
        return self.tags

    @property
    def truth(self):
        return self.g_truth


class TInstWithoutLogits:
    def __init__(self, batch_size, slen, tag_size):
        self.src_words = Variable(torch.LongTensor(batch_size, slen).zero_(), requires_grad=False)
        self.src_masks = Variable(torch.Tensor(batch_size, slen).zero_(), requires_grad=False)
        self.tags = Variable(torch.LongTensor(batch_size).zero_(), requires_grad=False)
        self.word_len = Variable(torch.LongTensor(batch_size).zero_(), requires_grad=False)

    def to_cuda(self, device):
        self.src_words = self.src_words.cuda(device)
        self.src_masks = self.src_masks.cuda(device)
        self.tags = self.tags.cuda(device)
        self.word_len = self.word_len.cuda(device)

    @property
    def inputs(self):
        return self.src_words, self.src_masks, self.word_len

    @property
    def targets(self):
        return self.tags

    @property
    def truth(self):
        return self.tags
