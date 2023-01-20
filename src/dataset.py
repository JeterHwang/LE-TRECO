import torch
from alphabets import Uniprot21
from torch.nn.utils.rnn import pad_sequence

class LSTMDataset:
    def __init__(self, seqs, alphabet=Uniprot21()):
        self.seqs = seqs
        self.alphabet = alphabet
        if not isinstance(self.alphabet, Uniprot21):
            self.batch_converter = self.alphabet.get_batch_converter()
            self.tokens = [seq[:1022] for seq in self.seqs]
        else:
            self.tokens = [torch.from_numpy(alphabet.encode(seq.encode('utf-8').upper())).long() for seq in self.seqs]
    
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.tokens[idx], idx
    
    def batch_sampler(self, toks_per_batch=4096):
        sizes = [(len(s), i) for i, s in enumerate(self.seqs)]
        sizes.sort()
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0
        
        for sz, i in sizes:
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)
        
        _flush_current_buf()
        return batches

    def collate_fn(self, samples):
        if hasattr(self, 'batch_converter'):
            seqs = [('A', sample[0]) for sample in samples]
            indices = [sample[1] for sample in samples]
            length = [len(seq) for seq in seqs]
            _, _, batch_tokens = self.batch_converter(seqs)
        else:
            seqs = [sample[0] for sample in samples]
            indices = [sample[1] for sample in samples]
            length = [len(seq) for seq in seqs]
            batch_tokens = pad_sequence(seqs, batch_first=True, padding_value=0)
        return batch_tokens, length, indices
