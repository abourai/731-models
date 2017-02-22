from collections import defaultdict
import dynet as dy
#import _gdynet as dy
import numpy as np
import random
import sys
import time
from itertools import count
from datetime import datetime

#dy.init()


## from EMNLP tutorial code
class Vocab:
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.iteritems()}
    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(count(0).next)
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)

    def size(self): return len(self.w2i.keys())

class Attention:
    def __init__(self, model, training_src, training_tgt, embed_size=500, hidden_size=500, attention_size=200):

        self.vw_src = Vocab.from_corpus(training_src)
        self.vw_tgt = Vocab.from_corpus(training_tgt)

        self.src_vocab_size = self.vw_src.size()
        self.tgt_vocab_size = self.vw_tgt.size()

        self.model = model
        self.training = [(x, y) for (x, y) in zip(training_src, training_tgt)]
        self.src_token_to_id, self.src_id_to_token = self.vw_src.w2i, self.vw_src.i2w
        self.tgt_token_to_id, self.tgt_id_to_token = self.vw_tgt.w2i, self.vw_tgt.i2w

        self.pad = '<S>'

        self.max_len = 80
        self.BATCH_SIZE = 100
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.layers = 1
        self.src_lookup = model.add_lookup_parameters((self.src_vocab_size, self.embed_size))
        self.tgt_lookup = model.add_lookup_parameters((self.tgt_vocab_size, self.embed_size))
        self.l2r_builder = dy.LSTMBuilder(self.layers, self.embed_size, self.hidden_size, model)
        self.r2l_builder = dy.LSTMBuilder(self.layers, self.embed_size, self.hidden_size, model)
    #    self.l2r_builder.set_dropout(0.5)
    #    self.r2l_builder.set_dropout(0.5)

        self.dec_builder = dy.LSTMBuilder(self.layers, 2 * self.hidden_size + self.embed_size , self.hidden_size, model)
    #    self.dec_builder.set_dropout(0.2)
        self.W_y = model.add_parameters((self.tgt_vocab_size, self.hidden_size))
        self.b_y = model.add_parameters((self.tgt_vocab_size))

        self.W1_att_f = model.add_parameters((self.attention_size, 2 * self.hidden_size))
        self.W1_att_e = model.add_parameters((self.attention_size, self.hidden_size))
        self.w2_att = model.add_parameters((self.attention_size))

    def restore(self,model,epoch_num):
        (self.src_lookup,self.tgt_lookup,self.l2r_builder,self.r2l_builder,self.dec_builder,self.W_y,self.b_y,
            self.W1_att_f,self.W1_att_e,self.w2_att) = model.load('attention.parameters-epoch-'+str(epoch_num))

    def save(self,epoch_num):
        self.model.save("attention.parameters-epoch-"+str(epoch_num),[self.src_lookup,self.tgt_lookup,self.l2r_builder,
            self.r2l_builder,self.dec_builder,self.W_y,self.b_y,self.W1_att_f,self.W1_att_e,self.w2_att])
        


    def __calc_attn_score(self, W1_att_f, W1_att_e, w2_att, h_fs_matrix, h_e):
        h_e_matrix = dy.concatenate_cols([h_e for i in range(h_fs_matrix.npvalue().shape[1])])
        layer_1 = dy.tanh(W1_att_f * h_fs_matrix + W1_att_e * h_e_matrix)

        return dy.transpose(layer_1) * w2_att

    # Calculates the context vector using a MLP
    # h_fs: matrix of embeddings for the source words
    # h_e: hidden state of the decoder
    def __attention_mlp(self, h_fs_matrix, h_e):
        W1_att_f = dy.parameter(self.W1_att_f)
        W1_att_e = dy.parameter(self.W1_att_e)
        w2_att = dy.parameter(self.w2_att)

        # Calculate the alignment score vector
        a_t = self.__calc_attn_score(W1_att_f, W1_att_e, w2_att, h_fs_matrix, h_e)
        alignment = dy.softmax(a_t)
        c_t = h_fs_matrix * alignment
        return c_t


    def __pad_batch(self, batch, src=True):

        max_batch_len = len(max(batch, key=len))

        padded_batch = []
        for sent in batch:

            if src:
                sent =  [self.src_token_to_id[self.pad]] *  (max_batch_len - len(sent)) + [self.src_token_to_id[x] for x in sent]
            else:
                sent =  [self.tgt_token_to_id[self.pad]] *  (max_batch_len - len(sent)) + [self.tgt_token_to_id[x] for x in sent]
            padded_batch.append(sent)
        return padded_batch

    def __mask(self, batch):

        masks = []
        max_len = len(max(batch, key=len))
        for padded_sent in batch:
            len_sent = len(padded_sent)

            masks.append([0] * (max_len - len_sent) + [1] * len_sent)
        return masks

    # Training step over a sentence batch
    def step_batch(self, instances):
        dy.renew_cg()

        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)

        src_sents = [x[0] for x in instances]
        padded_src = self.__pad_batch(src_sents)
        masks_src = np.transpose(self.__mask(padded_src))

        src_cws = np.transpose(padded_src)

        tgt_sents = [x[1] for x in instances]
        tgt_ids = []

        for sent in tgt_sents:
            sent = [self.tgt_token_to_id[x] for x in sent]
            tgt_ids.append(sent)

        tgt_ids = map(list, zip(*tgt_ids))

        padded_src_rev = list(reversed(padded_src))
        src_cws_rev = np.transpose(padded_src_rev)

        # Bidirectional representations
        l2r_state = self.l2r_builder.initial_state()
        r2l_state = self.r2l_builder.initial_state()
        l2r_contexts = []
        r2l_contexts = []
        for (cws_l2r, cws_r2l) in zip(src_cws, src_cws_rev):

            l2r_state = l2r_state.add_input(dy.lookup_batch(self.src_lookup, cws_l2r))
            r2l_state = r2l_state.add_input(dy.lookup_batch(self.src_lookup, cws_r2l))
            l2r_contexts.append(l2r_state.output()) #[<S>, x_1, x_2, ..., </S>]
            r2l_contexts.append(r2l_state.output()) #[</S> x_n, x_{n-1}, ... <S>]

        r2l_contexts.reverse() #[<S>, x_1, x_2, ..., </S>]
        # Combine the left and right representations for every word
        h_fs = []
        for (l2r_i, r2l_i) in zip(l2r_contexts, r2l_contexts):
            h_fs.append(dy.concatenate([l2r_i, r2l_i]))
        h_fs_matrix = dy.concatenate_cols(h_fs)
        losses = []
        num_words = 0

        # Decoder
        c_t = dy.vecInput(self.hidden_size * 2)
        start = dy.concatenate([dy.lookup_batch(self.tgt_lookup, len(tgt_sents) * [self.tgt_token_to_id['<S>']]), c_t])
        dec_state = self.dec_builder.initial_state().add_input(start)

        for (cws, nws, mask) in zip(tgt_ids, tgt_ids[1:], masks_src):
            h_e = dec_state.output()
            c_t = self.__attention_mlp(h_fs_matrix, h_e)

            # Get the embedding for the current target word
            embed_t = dy.lookup_batch(self.tgt_lookup, cws)

            # Create input vector to the decoder
            x_t = dy.concatenate([embed_t, c_t])
            dec_state = dec_state.add_input(x_t)

            y_star = b_y + W_y * dec_state.output()
            loss = dy.pickneglogsoftmax_batch(y_star, nws)#-dy.log(dy.pick(y_star, nws))

            if mask[0] == 0:
                mask_loss = dy.reshape(dy.inputVector(mask), (1,), self.BATCH_SIZE)
                loss = loss * mask_loss

            losses.append(loss)
            num_words += 1

        return dy.sum_batches(dy.esum(losses)), num_words

    def translate_sentence(self, sent):
        dy.renew_cg()

        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)

        sent_rev = list(reversed(sent))
        # Bidirectional representations
        l2r_state = self.l2r_builder.initial_state()
        r2l_state = self.r2l_builder.initial_state()
        l2r_contexts = []
        r2l_contexts = []
        for (cw_l2r, cw_r2l) in zip(sent, sent_rev):
            l2r_state = l2r_state.add_input(dy.lookup(self.src_lookup, self.src_token_to_id[cw_l2r]))
            r2l_state = r2l_state.add_input(dy.lookup(self.src_lookup, self.src_token_to_id[cw_r2l]))
            l2r_contexts.append(l2r_state.output())
            r2l_contexts.append(r2l_state.output())
        r2l_contexts.reverse()

        h_fs = []
        for (l2r_i, r2l_i) in zip(l2r_contexts, r2l_contexts):
            h_fs.append(dy.concatenate([l2r_i, r2l_i]))
        h_fs_matrix = dy.concatenate_cols(h_fs)

        # Decoder
        trans_sentence = ['<S>']
        cw = trans_sentence[-1]
        c_t = dy.vecInput(self.hidden_size * 2)
        start = dy.concatenate([dy.lookup(self.tgt_lookup, self.tgt_token_to_id['<S>']), c_t])
        dec_state = self.dec_builder.initial_state().add_input(start)
        while len(trans_sentence) < self.max_len:
            h_e = dec_state.output()
            c_t = self.__attention_mlp(h_fs_matrix, h_e)
            embed_t = dy.lookup(self.tgt_lookup, self.tgt_token_to_id[cw])
            x_t = dy.concatenate([embed_t, c_t])
            dec_state = dec_state.add_input(x_t)
            y_star = b_y + W_y * dec_state.output()
            p = dy.softmax(y_star)
            cw = self.tgt_id_to_token[np.argmax(p.npvalue())]
            if cw == '</S>':
                break
            trans_sentence.append(cw)
        return ' '.join(trans_sentence[1:])

def read_file(file_name):
    with open(file_name,'r') as train_file:
        ori_sentences = [['<S>']+sentence.split()+['</S>'] for sentence in train_file]
    return ori_sentences


def main():
    training_log = open('training-'+str(datetime.now())+'.log','w')
    model = dy.Model()
    trainer = dy.SimpleSGDTrainer(model)
    training_src = read_file(sys.argv[1])
    training_tgt = read_file(sys.argv[2])
    dev_src = read_file(sys.argv[3])
    dev_tgt = read_file(sys.argv[4])
    test_src = read_file(sys.argv[5])
    attention = Attention(model, list(training_src), list(training_tgt))

    train_data = zip(training_src, training_tgt)
    train_data.sort(key=lambda x: -len(x[0]))

    train_src = [sent[0] for sent in train_data]
    train_tgt = [sent[1] for sent in train_data]

    start = time.time()
    for epoch in range(5000):
            epoch_loss = 0
            train_zip = zip(train_src, train_tgt)
            i = 0
            while i < len(train_zip):
                esum,num_words = attention.step_batch(train_zip[i:i+attention.BATCH_SIZE])
                i += attention.BATCH_SIZE
                epoch_loss += esum.scalar_value() / num_words
                esum.backward()
                trainer.update()
            # if epoch_loss < 10:
            #     end = time.time()
            #     print 'TIME ELAPSED:', end - start, 'SECONDS'
            #     break
            print 'Epoch:',epoch
            training_log.write("Epoch %d: loss=%f \n" % (epoch, epoch_loss))
            training_log.flush()
            trainer.update_epoch(1.0)
            training_log.write(attention.translate_sentence(training_src[0])+'\n')
            if epoch % 100 == 0:
                attention.save(epoch)
    # new_model = dy.Model()
    # new_attention = Attention(new_model, list(training_src), list(training_tgt))
    # new_attention.restore(model,10)
    # print new_attention.translate_sentence(training_src[0])
            

if __name__ == '__main__': main()