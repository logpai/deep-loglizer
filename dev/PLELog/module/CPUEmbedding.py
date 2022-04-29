from CONSTANTS import *

from torch.nn.parameter import Parameter


class CPUEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super(CPUEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def _apply(self, fn):
        str_func = str(fn)
        print(str_func)
        if 'cuda.<locals>.<lambda>' in str_func:
            print('Always in cpu: function disabled')
            return self

        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Tensors stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        return self

    def cuda(self, device=None):
        print('Always in cpu')
        return self.cpu()

    def forward(self, input):
        if input.is_cuda:
            device = input.get_device()
            input = input.cpu()
            output = F.embedding(input, self.weight, self.padding_idx)
            return output.cuda(device)
        else:
            try:
                output = F.embedding(input, self.weight, self.padding_idx)
            except IndexError:
                print(input)
            return output

    def extra_repr(self):
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        return s.format(**self.__dict__)
