from collections import defaultdict
import dynet as dy
import _gdynet as dy
import numpy as np
import random
import sys
import time
from itertools import count
from datetime import datetime

dy.init()

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
    def __init__(self, model, training_src, training_tgt, embed_size=256, hidden_size=256, attention_size=32):

        
        self.vw_src = Vocab.from_corpus(training_src)
        self.vw_tgt = Vocab.from_corpus(training_tgt)

        self.src_vocab_size = self.vw_src.size()
        self.tgt_vocab_size = self.vw_tgt.size()

        self.model = model
        self.training = [(x, y) for (x, y) in zip(training_src, training_tgt)]
        self.src_token_to_id, self.src_id_to_token = self.vw_src.w2i, self.vw_src.i2w
        self.tgt_token_to_id, self.tgt_id_to_token = self.vw_tgt.w2i, self.vw_tgt.i2w

        self.src_pad = '<S>'
        self.tgt_pad = '</S>'
        self.beam_size = 10
        self.max_len = 80
        self.BATCH_SIZE = 32
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.layers = 1
        self.src_lookup = model.add_lookup_parameters((self.src_vocab_size, self.embed_size))
        self.tgt_lookup = model.add_lookup_parameters((self.tgt_vocab_size, self.embed_size))
        self.l2r_builder = dy.LSTMBuilder(self.layers, self.embed_size, self.hidden_size, model)
        self.r2l_builder = dy.LSTMBuilder(self.layers, self.embed_size, self.hidden_size, model)
        self.l2r_builder.set_dropout(0.2)
        self.r2l_builder.set_dropout(0.2)

        self.dec_builder = dy.LSTMBuilder(self.layers, 2 * self.hidden_size + self.embed_size , self.hidden_size, model)
        self.dec_builder.set_dropout(0.2)
        self.W_y = model.add_parameters((self.tgt_vocab_size, self.hidden_size))
        self.b_y = model.add_parameters((self.tgt_vocab_size))

        self.W1_att_f = model.add_parameters((self.attention_size, 2 * self.hidden_size))
        self.W1_att_e = model.add_parameters((self.attention_size, self.hidden_size))
        self.w2_att = model.add_parameters((self.attention_size))

    def restore(self):
        (self.src_lookup,self.tgt_lookup,self.l2r_builder,self.r2l_builder,self.dec_builder,self.W_y,self.b_y,
            self.W1_att_f,self.W1_att_e,self.w2_att) = self.model.load('attention.parameters')

    def save(self):
        self.model.save("attention.parameters",[self.src_lookup,self.tgt_lookup,self.l2r_builder,
            self.r2l_builder,self.dec_builder,self.W_y,self.b_y,self.W1_att_f,self.W1_att_e,self.w2_att])
          


    def __calc_attn_score(self, W1_att_f, W1_att_e, w2_att, h_fs_matrix, h_e):
        return dy.transpose(dy.tanh(dy.colwise_add( W1_att_f * h_fs_matrix,W1_att_e * h_e))) * w2_att

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


    def __pad_batch(self, batch, src):

        max_batch_len = len(max(batch, key=len))
        padded_batch = []
        for sent in batch:
            if src:
                sent =  [self.src_token_to_id[x] for x in sent]
            else:
                sent =  [self.tgt_token_to_id[x] for x in sent] + [self.tgt_token_to_id[self.tgt_pad]] * (max_batch_len - len(sent))
            padded_batch.append(sent)
        return padded_batch

    def __mask(self, batch):
        masks = []
        max_len = len(max(batch, key=len))
        num_words = 0
        for padded_sent in batch:
            len_sent = len(padded_sent)
            num_words += len_sent
            masks.append([1] * len_sent + [0] * (max_len - len_sent))
        return masks, num_words

    # Training step over a sentence batch
    def step_batch(self, instances):
        dy.renew_cg()
        self.l2r_builder.set_dropout(0.2)
        self.r2l_builder.set_dropout(0.2)
        self.dec_builder.set_dropout(0.2)
    
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        src_sents = [x[0] for x in instances]
        padded_src = self.__pad_batch(src_sents, True)
        src_cws = np.transpose(padded_src)
        tgt_sents = [x[1] for x in instances]
        padded_tgt = self.__pad_batch(tgt_sents, False)
        masks_tgt, num_words = self.__mask(tgt_sents)
        masks_tgt = np.transpose(masks_tgt)
        padded_tgt = np.transpose(padded_tgt)
        instance_size = len(instances)
        src_cws_rev = list(reversed(src_cws))
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
        
        # Decoder
        c_t = dy.vecInput(self.hidden_size * 2)
        start = dy.concatenate([dy.lookup_batch(self.tgt_lookup, len(tgt_sents) * [self.tgt_token_to_id[self.src_pad]]), c_t])
        dec_state = self.dec_builder.initial_state().add_input(start)

        for (cws, nws, mask) in zip(padded_tgt, padded_tgt[1:], masks_tgt):
            h_e = dec_state.output()
            c_t = self.__attention_mlp(h_fs_matrix, h_e)
            # Get the embedding for the current target word
            embed_t = dy.lookup_batch(self.tgt_lookup, cws)
            # Create input vector to the decoder
            x_t = dy.concatenate([embed_t, c_t])
            dec_state = dec_state.add_input(x_t)
            y_star = b_y + W_y * dec_state.output()
            loss = dy.pickneglogsoftmax_batch(y_star, nws)
            if mask[-1] == 0:
                mask_loss = dy.reshape(dy.inputVector(mask), (1,), instance_size)
                masked = loss * mask_loss
                losses.append(masked)
            else:
                losses.append(loss)
            #losses = [(x / num_words) for x in losses]
        return dy.sum_batches(dy.esum(losses)), num_words

    def translate_sentence_beam(self, sent):
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
        cws = []
        c_t = dy.vecInput(self.hidden_size * 2)
        start = dy.concatenate([dy.lookup(self.tgt_lookup, self.tgt_token_to_id[self.src_pad]), c_t])
        dec_state = self.dec_builder.initial_state().add_input(start)
        sentence_dict = defaultdict(lambda : defaultdict(list))
        end_sign = self.tgt_token_to_id[self.tgt_pad]
        #first_iter 
        h_e = dec_state.output()
        c_t = self.__attention_mlp(h_fs_matrix, h_e)
        embed_t = dy.lookup(self.tgt_lookup, self.tgt_token_to_id[self.src_pad])
        x_t = dy.concatenate([embed_t, c_t])
        dec_state = dec_state.add_input(x_t)
        y_star = (b_y + W_y * dec_state.output())
        p = dy.log(dy.softmax(y_star)).npvalue()
        cws = np.argpartition(-p,self.beam_size)[:self.beam_size]
        history_path = p[cws]
        
        trans_iter = 0
        for i in range(self.beam_size):
            sentence_dict[trans_iter][i] = [self.tgt_id_to_token[cws[i]]]
        try:
            index = np.where(cws==end_sign)[0][0]
           # print index,'sentence ends normaly.'
            return ''
        except :
            #print 'no </S>'
            pass
        #print 'first dict',sentence_dict
        while trans_iter < self.max_len:
            h_e = dec_state.output()
            c_t = self.__attention_mlp(h_fs_matrix, h_e)
            y_stars = np.zeros([self.tgt_vocab_size,self.beam_size])
            #print 'before y_stars.shape',y_stars.shape
            embed_t = dy.lookup_batch(self.tgt_lookup, cws)
            x_t = dy.concatenate([embed_t, c_t])
            dec_state = dec_state.add_input(x_t)
            #print 'element',(b_y + W_y * dec_state.output()).npvalue().shape
            y_stars = dy.log(dy.softmax(b_y + W_y * dec_state.output())).npvalue()
            #print 'y_stars.shape',y_stars.shape
            #print 'history_path',history_path.shape
            #print 'beam',self.beam_size,self.tgt_vocab_size
            #print 'y_stars_before',y_stars.shape,y_stars
            #print 'history_path',history_path.shape, history_path
            y_stars += history_path
            #print 'y_stars_after',y_stars.shape,y_stars
            y_star_all = np.reshape(y_stars.T,(self.beam_size * self.tgt_vocab_size,))
            #print 'y_star_all',y_star_all.shape
            #print 'y_star_all_sort',np.sort(y_star_all)
            beam_indexes = np.argpartition(-y_star_all,self.beam_size)[:self.beam_size]
            beam_sentence_index = list(map( (lambda x: x/self.tgt_vocab_size),beam_indexes))
            #end_sign = np.array([self.tgt_token_to_id['</S>'],self.tgt_token_to_id['</S>']+self.tgt_vocab_size,self.tgt_token_to_id['</S>']+2*self.tgt_vocab_size,self.tgt_token_to_id['</S>']+3*self.tgt_vocab_size])
            cws = [index % self.tgt_vocab_size for index in beam_indexes]
            history_path = y_star_all[beam_indexes]       
            # print 'beam_indexes ',beam_indexes, y_star_all[beam_indexes] #y_star_all[end_sign]
            # print 'beam_sentence_index',beam_sentence_index
            # print 'history_path',history_path
            # print 'cws words',cws,end_sign
            trans_iter += 1
            #print trans_iter -1 ,'dict',sentence_dict
            max_score = -np.inf
            find_end_sign = False
            for i,cw in enumerate(cws):
                if cw == end_sign:
                    find_end_sign = True
                    if max_score < history_path[i]:
                        max_score = history_path[i]
                        max_index = i 
            if find_end_sign:
                #print 'return',trans_iter-1,max_index, beam_sentence_index[max_index]
                return ' '.join(sentence_dict[trans_iter-1][beam_sentence_index[max_index]])
            for i in range(self.beam_size):
                sentence_dict[trans_iter][i] = sentence_dict[trans_iter - 1][beam_sentence_index[i]] + [self.tgt_id_to_token[cws[i]]]
            #print trans_iter,'dict',sentence_dict
            #index = cws.index(self.tgt_token_to_id['</S>'])
        #print 'sentence too long'
        return ' '.join(sentence_dict[trans_iter][np.argmax(history_path)])

    def translate_sentence_ori(self, sent):
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
        trans_sentence = [self.src_pad]
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
            p = dy.log(dy.softmax(y_star))
            #print p.npvalue()
            cw = self.tgt_id_to_token[np.argmax(p.npvalue())]
            #print 'ori_cw',cw, p.npvalue()[np.argmax(p.npvalue())]
            if cw == self.tgt_pad:
                break
            trans_sentence.append(cw)
        return ' '.join(trans_sentence[1:])

    def debug(self,cws):
        str1 = ''
        for cw in cws:
            str1 += self.tgt_id_to_token[cw] + ' '
        return str1

def read_file(file_name):
    with open(file_name,'r') as train_file:
        ori_sentences = [['<S>']+sentence.strip().split(' ')+['</S>'] for sentence in train_file]
    return ori_sentences


def sentence_clean(sentences,word_dict):
    for sentence in sentences:
        for i,word in enumerate(sentence):
            if word_dict[word] <= 2:
                sentence[i] = '<unknown>'
    return sentences

def build_dict(sentences):
    word_freq = defaultdict(lambda: 0)
    for sentence in sentences:
        for word in sentence:
            word_freq[word] += 1
    return word_freq
def main():
    training_log = open('training-'+str(datetime.now())+'.log','w')
    translation_result = open('translation-beam.txt','w')
    translation_result1 = open('translation-greedy.txt','w')
    
    model = dy.Model()
    trainer = dy.AdamTrainer(model)
    training_src = read_file(sys.argv[1])
    word_freq_src = build_dict(training_src)

    training_tgt = read_file(sys.argv[2])
    word_freq_tgt = build_dict(training_tgt)

    training_src = sentence_clean(training_src, word_freq_src)
    training_tgt = sentence_clean(training_tgt, word_freq_tgt) 
    dev_src = sentence_clean(read_file(sys.argv[3]),word_freq_src)
    dev_tgt = sentence_clean(read_file(sys.argv[4]),word_freq_tgt)
    test_src = sentence_clean(read_file(sys.argv[5]),word_freq_src)
    attention = Attention(model, training_src, training_tgt)
    
    train_data = zip(training_src, training_tgt)
    train_data.sort(key=lambda x: -len(x[0]))

    train_src = [sent[0] for sent in train_data]
    train_tgt = [sent[1] for sent in train_data]
    train_zip = zip(train_src, train_tgt)
    train_final = defaultdict(list)
    for train in train_zip:
        train_final[len(train[0])].append(train)
    
    dev_data = zip(dev_src, dev_tgt)
    dev_data.sort(key=lambda x: -len(x[0]))
    
    dev_final = defaultdict(list)

    for dev in dev_data:
        dev_final[len(dev[0])].append(dev)

    start = time.time()
    print 'BEGINNING TRAINING'
    min_dev_loss = float('inf')
    
    for epoch in range(25):
        epoch_loss = 0
        total_words = 0
        for len_sent, sents in train_final.items():
            i = 0
            len_batch = len(sents)
            while i < len_batch:
                esum,num_words = attention.step_batch(sents[i:i+attention.BATCH_SIZE])
                i += attention.BATCH_SIZE
                total_words += num_words
                epoch_loss += esum.scalar_value() #/ num_words
                esum.backward()
                trainer.update()
        print 'Epoch:',epoch, 'Perplexity Per Word:', np.exp(epoch_loss / total_words)
        training_log.write("Epoch %d: perplexity=%f \n" % (epoch, np.exp(epoch_loss / total_words)))
        training_log.flush()

        dev_loss = 0
        dev_words = 0
        for len_sent, sents in dev_final.items():
            i = 0
            len_batch = len(sents)
            while i < len_batch:
                esum,num_words = attention.step_batch(sents[i:i+attention.BATCH_SIZE])
                i += attention.BATCH_SIZE
                dev_words += num_words
                dev_loss += esum.scalar_value() #/ num_words
        print 'Dev:',epoch, 'Perplexity Per Word:', np.exp(dev_loss / dev_words)
        training_log.write("Dev %d: perplexity=%f \n" % (np.exp(epoch, dev_loss / dev_words)))
        training_log.flush()
        if np.exp(dev_loss / dev_words) < minDevLoss:
            attention.save()
            minDevLoss = np.exp(dev_loss / dev_words)
    


    for sentence in test_src:
        translation_result1.write(attention.translate_sentence_ori(sentence)+'\n')
        translation_result1.flush()
        translation_result.write(attention.translate_sentence_beam(sentence)+'\n')
        translation_result.flush()
        
    # new_model = dy.Model()
    # new_attention = Attention(new_model, list(training_src), list(training_tgt))
    # new_attention.restore(model,10)
    # print new_attention.translate_sentence(training_src[0])
            
if __name__ == '__main__': main()
