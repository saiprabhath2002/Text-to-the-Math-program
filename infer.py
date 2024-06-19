import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchtext.vocab as vocab
from torchtext.vocab import GloVe
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import time
import torch.optim as optim
import matplotlib.pyplot as plt
from ordered_set import OrderedSet
import random
from transformers import BertTokenizer
from transformers import BertModel
import argparse



def modify_and_overwrite_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    modified_data = []
    for item in data:
        modified_item = {
            "Problem": item["Problem"],
            "linear_formula": ""
        }
        modified_data.append(modified_item)
    with open(file_path, 'w') as file:
        json.dump(modified_data, file, indent=4)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

special_tags=["<sos>","<eos>","<unk>","<pad>"]
symbols = ["(", ")", ",","|","const_"]
symbols.extend(range(10))
pos_special_tokens_prob=[0,0,0,0]
pos_special_tokens_sol=[0,0,0,0]
embed_dim=100
beam_size=10

def collate_c_d(batch):
    
    max_len_problem = max([len(sample[0]) for sample in batch])
    max_len_solution = max([len(sample[1]) for sample in batch])
    
    padded_prob = torch.empty((len(batch), max_len_problem), dtype=torch.long)
    padded_prob.fill_(0)
    padded_sol = torch.empty((len(batch), max_len_solution), dtype=torch.long)
    padded_sol.fill_(pos_special_tokens_sol[3])
    prob_attn_mask = torch.zeros((len(batch), max_len_problem), dtype=torch.long)
    for idx in range(len(batch)):
        prob_len = len(batch[idx][0])
        padded_prob[idx, :len(batch[idx][0])] = torch.LongTensor(batch[idx][0])
        padded_sol[idx, :len(batch[idx][1])] = torch.LongTensor(batch[idx][1])
        prob_attn_mask[idx, :prob_len] = torch.ones((1, prob_len), dtype=torch.long)
    return (padded_prob,padded_sol,prob_attn_mask)

class load_data_train_c_d(Dataset):
    def __init__(self,json_path):
        self.path=json_path
        self.data=[]
        self.loaddata()
        self.problem_unique_words,self.sol_unique_words=self.gen_all_unique_words()
        self.problem_word2int = {word: i for i, word in enumerate(self.problem_unique_words)}
        self.problem_int2word = {i: word for word, i in self.problem_word2int.items()}
        self.sol_word2int = {word: i for i, word in enumerate(self.sol_unique_words)}
        self.sol_int2word = {i: word for word, i in self.sol_word2int.items()}
        self.max_problem_len=self.get_max_len()
        self.get_special_pos_prob()
        self.get_special_pos_sol()
        self.en_tokenizer =  BertTokenizer.from_pretrained("bert-base-cased")
    def get_special_pos_prob(self):
        for i,t in enumerate(special_tags):
            pos_special_tokens_prob[i]=self.problem_word2int[t]
            
    def get_special_pos_sol(self):
        for i,t in enumerate(special_tags):
            pos_special_tokens_sol[i]=self.sol_word2int[t]
            
    def gen_all_unique_words(self):
        u1=set(special_tags)
        u2=set(symbols+special_tags)
        for i,(prob,sol) in enumerate(self.data):
            for word in prob.split():
                u1.add(word)
            operations = sol.split("|")
            for operation in operations:
                if not operation:  
                    continue
                operation_name = operation.split("(")[0]
                u2.add(operation_name) 
                content = operation[operation.find("(")+1:operation.find(")")]
                tokens = content.split(",")
                u2.update(tokens)
        return u1,u2
    
    def tokanize_problem(self,i):
#         return self.data[i][0].split()
        return self.en_tokenizer.encode(self.data[i][0])
    
    def tokanize_sol(self,i):
        l=[]
        operations = self.data[i][1].split("|")
        for j,operation in enumerate(operations):
            if not operation:  
                continue
            operation_name = operation.split("(")[0]
            l.append(operation_name) 
            l.append("(")
            content = operation[operation.find("(")+1:operation.find(")")]
            tokens = content.split(",")
            new=[]
            for i, token in enumerate(tokens):
                new.append(token)
                if i < len(tokens) - 1:
                    new.append(",")
            l.extend(new)
            l.append(")")
            if j < len(operations) - 1:
                    l.append("|")
        return l
    
    def get_max_len(self):
        m=0
        for (p,s) in self.data:
            for p1 in p:
                m=max(m,len(p1.split()))
        return m

    def loaddata(self):
        with open(self.path, 'r') as f:
            data = json.load(f)
            for i in data:
                #list(i["Problem"].split())
#                 p=str(i["Problem"]).split()
                self.data.append((i["Problem"],i["linear_formula"]))
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        problem=self.tokanize_problem(i)
        sol=["<sos>"]+self.tokanize_sol(i)+["<eos>"]
#         problem = [self.problem_word2int[q] if q in self.problem_word2int else self.problem_word2int["<unk>"] for q in problem]
        sol = [self.sol_word2int[q] if q in self.sol_word2int else self.sol_word2int["<unk>"] for q in sol]
        return ((problem,sol))

class load_data_test_c_d(Dataset):
    def __init__(self,json_path,train):
        self.path=json_path
        self.data=[]
        self.loaddata()
        self.problem_unique_words,self.sol_unique_words=train.problem_unique_words,train.sol_unique_words
        self.problem_word2int = train.problem_word2int
        self.problem_int2word = train.problem_int2word
        self.sol_word2int = train.sol_word2int
        self.sol_int2word =train.sol_int2word
        self.en_tokenizer =  BertTokenizer.from_pretrained("bert-base-cased")
    def tokanize_problem(self,i):
        return self.en_tokenizer.encode(self.data[i][0])
    
    def tokanize_sol(self,i):
        l=[]
        operations = self.data[i][1].split("|")
        for j,operation in enumerate(operations):
            if not operation:  
                continue
            operation_name = operation.split("(")[0]
            l.append(operation_name) 
            l.append("(")
            content = operation[operation.find("(")+1:operation.find(")")]
            tokens = content.split(",")
            new=[]
            for i, token in enumerate(tokens):
                new.append(token)
                if i < len(tokens) - 1:
                    new.append(",")
            l.extend(new)
            l.append(")")
            if j < len(operations) - 1:
                    l.append("|")
        return l

    def loaddata(self):
        with open(self.path, 'r') as f:
            data = json.load(f)
            for i in data:
                #list(i["Problem"].split())
#                 p=str(i["Problem"]).split()
                self.data.append((i["Problem"],i["linear_formula"]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        problem=self.tokanize_problem(i)
        sol=["<sos>"]+self.tokanize_sol(i)+["<eos>"]
#         problem = [self.problem_word2int[q] if q in self.problem_word2int else self.problem_word2int["<unk>"] for q in problem]
        sol = [self.sol_word2int[q] if q in self.sol_word2int else self.sol_word2int["<unk>"] for q in sol]
        return ((problem,sol))

class BERT_Encoder_d(nn.Module):
    def __init__(self, fine_tune_layers=2):
        super(BERT_Encoder_d, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')

        for param in self.bert.parameters():
            param.requires_grad = False
        if fine_tune_layers > 0:
            for layer in self.bert.encoder.layer[-fine_tune_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
    

class Attention_d(nn.Module):
    def __init__(self, enc_hid_dim=768, dec_hid_dim=128):
        super(Attention_d, self).__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, encoder_outputs, hidden):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1) 
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2))) 
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class LSTM_Decoder_d(nn.Module):
    def __init__(self, output_dim, emb_dim=200, enc_hid_dim=768, dec_hid_dim=128, dropout=0.5):
        super(LSTM_Decoder_d, self).__init__()
        self.attention = Attention_d()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + enc_hid_dim, dec_hid_dim, batch_first=True)
        self.fc_out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        embedded = self.dropout(self.embedding(input.unsqueeze(1)))  
        attn_weighted = self.attention(encoder_outputs, hidden[0])  
        attn_weighted = attn_weighted.unsqueeze(1) 
        weighted = torch.bmm(attn_weighted, encoder_outputs)  
        rnn_input = torch.cat((embedded, weighted), dim=2)  
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2).squeeze(1))
        return prediction, hidden, cell

class Seq2Seq_d(nn.Module):
    def __init__(self,max_len_pred=None,device=device):
        super(Seq2Seq_d, self).__init__()
        self.encoder = BERT_Encoder_d()
        self.decoder = LSTM_Decoder_d(output_dim=max_len_pred)
        self.device = device

    def forward(self, src,trg, src_mask,  teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.embedding.num_embeddings
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs = self.encoder(src, src_mask)
        hidden = torch.zeros(1, batch_size, self.decoder.rnn.hidden_size).to(self.device)
        cell = torch.zeros(1, batch_size, self.decoder.rnn.hidden_size).to(self.device)
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        predicted_tokens = outputs.argmax(2) 
        return outputs,predicted_tokens


class BERT_Encoder_c(nn.Module):
    def __init__(self):
        super(BERT_Encoder_c, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state 
    

class Attention_c(nn.Module):
    def __init__(self, enc_hid_dim=768, dec_hid_dim=128):
        super(Attention_c, self).__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, encoder_outputs, hidden):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2))) 
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)

class LSTM_Decoder_c(nn.Module):
    def __init__(self, output_dim, emb_dim=200, enc_hid_dim=768, dec_hid_dim=128, dropout=0.5):
        super(LSTM_Decoder_c, self).__init__()
        self.attention = Attention_c()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim + enc_hid_dim, dec_hid_dim, batch_first=True)
        self.fc_out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        embedded = self.dropout(self.embedding(input.unsqueeze(1))) 
        attn_weighted = self.attention(encoder_outputs, hidden[0])  
        attn_weighted = attn_weighted.unsqueeze(1) 
        weighted = torch.bmm(attn_weighted, encoder_outputs)  
        rnn_input = torch.cat((embedded, weighted), dim=2)  
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=2).squeeze(1))
        return prediction, hidden, cell

class Seq2Seq_c(nn.Module):
    def __init__(self,max_len_pred=None,device=device):
        super(Seq2Seq_c, self).__init__()
        self.encoder = BERT_Encoder_c()
        self.decoder = LSTM_Decoder_c(output_dim=max_len_pred)
        self.device = device

    def forward(self, src,trg, src_mask,  teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.embedding.num_embeddings
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs = self.encoder(src, src_mask)
        hidden = torch.zeros(1, batch_size, self.decoder.rnn.hidden_size).to(self.device)
        cell = torch.zeros(1, batch_size, self.decoder.rnn.hidden_size).to(self.device)
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        predicted_tokens = outputs.argmax(2) 
        return outputs,predicted_tokens

def collate_a_b(batch):
    
    max_len_problem = max([len(sample[0]) for sample in batch])
    max_len_solution = max([len(sample[1]) for sample in batch])
    
    padded_prob = torch.empty((len(batch), max_len_problem), dtype=torch.long)
    padded_prob.fill_(pos_special_tokens_prob[3])
    padded_sol = torch.empty((len(batch), max_len_solution), dtype=torch.long)
    padded_sol.fill_(pos_special_tokens_sol[3])

    for idx in range(len(batch)):
        
        padded_prob[idx, :len(batch[idx][0])] = torch.LongTensor(batch[idx][0])
        padded_sol[idx, :len(batch[idx][1])] = torch.LongTensor(batch[idx][1])
        
    return (padded_prob,padded_sol)


class load_data_train_a_b(Dataset):
    def __init__(self,json_path):
        self.path=json_path
        self.data=[]
        self.loaddata()
        self.problem_unique_words,self.sol_unique_words=self.gen_all_unique_words()
        self.problem_word2int = {word: i for i, word in enumerate(self.problem_unique_words)}
        self.problem_int2word = {i: word for word, i in self.problem_word2int.items()}
        self.sol_word2int = {word: i for i, word in enumerate(self.sol_unique_words)}
        self.sol_int2word = {i: word for word, i in self.sol_word2int.items()}
        self.max_problem_len=self.get_max_len()
        self.get_special_pos_prob()
        self.get_special_pos_sol()
    def get_special_pos_prob(self):
        for i,t in enumerate(special_tags):
            pos_special_tokens_prob[i]=self.problem_word2int[t]
            
    def get_special_pos_sol(self):
        for i,t in enumerate(special_tags):
            pos_special_tokens_sol[i]=self.sol_word2int[t]
            
    def gen_all_unique_words(self):
        u1=OrderedSet(special_tags)
        u2=OrderedSet(symbols+special_tags)
        for i,(prob,sol) in enumerate(self.data):
            for word in prob.split():
                u1.add(word)
            operations = sol.split("|")
            l=self.tokanize_sol(i)
            for t in l:
                u2.add(t)
        return u1,u2
    
    def tokanize_problem(self,i):
        return self.data[i][0].split()
    
    def tokanize_sol(self,i):
        l=[]
        operations = self.data[i][1].split("|")
        for j,operation in enumerate(operations):
            if not operation:  
                continue
            operation_name = operation.split("(")[0]
            l.append(operation_name) 
            l.append("(")
            content = operation[operation.find("(")+1:operation.find(")")]
            tokens = content.split(",")
            new=[]
            for i, token in enumerate(tokens):
                new.append(token)
                if i < len(tokens) - 1:
                    new.append(",")
            l.extend(new)
            l.append(")")
            if j < len(operations) - 1:
                    l.append("|")
        return l
    
    def get_max_len(self):
        m=0
        for (p,s) in self.data:
            for p1 in p:
                m=max(m,len(p1.split()))
        return m

    def loaddata(self):
        with open(self.path, 'r') as f:
            data = json.load(f)
            for i in data:
                #list(i["Problem"].split())
#                 p=str(i["Problem"]).split()
                self.data.append((i["Problem"],i["linear_formula"]))
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        problem=["<sos>"]+self.tokanize_problem(i)+["<eos>"]
        sol=["<sos>"]+self.tokanize_sol(i)+["<eos>"]
        problem = [self.problem_word2int[q] if q in self.problem_word2int else self.problem_word2int["<unk>"] for q in problem]
        sol = [self.sol_word2int[q] if q in self.sol_word2int else self.sol_word2int["<unk>"] for q in sol]
        return ((problem,sol))


class load_data_test_a_b(Dataset):
    def __init__(self,json_path,train):
        self.path=json_path
        self.data=[]
        self.loaddata()
        self.problem_unique_words,self.sol_unique_words=train.problem_unique_words,train.sol_unique_words
        self.problem_word2int = train.problem_word2int
        self.problem_int2word = train.problem_int2word
        self.sol_word2int = train.sol_word2int
        self.sol_int2word =train.sol_int2word
    
    def tokanize_problem(self,i):
        return self.data[i][0].split()
    
    def tokanize_sol(self,i):
        l=[]
        operations = self.data[i][1].split("|")
        for j,operation in enumerate(operations):
            if not operation:  
                continue
            operation_name = operation.split("(")[0]
            l.append(operation_name) 
            l.append("(")
            content = operation[operation.find("(")+1:operation.find(")")]
            tokens = content.split(",")
            new=[]
            for i, token in enumerate(tokens):
                new.append(token)
                if i < len(tokens) - 1:
                    new.append(",")
            l.extend(new)
            l.append(")")
            if j < len(operations) - 1:
                    l.append("|")
        return l

    def loaddata(self):
        with open(self.path, 'r') as f:
            data = json.load(f)
            for i in data:
                #list(i["Problem"].split())
#                 p=str(i["Problem"]).split()
                self.data.append((i["Problem"],i["linear_formula"]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        problem=["<sos>"]+self.tokanize_problem(i)+["<eos>"]
        sol=["<sos>"]+self.tokanize_sol(i)+["<eos>"]
        problem = [self.problem_word2int[q] if q in self.problem_word2int else self.problem_word2int["<unk>"] for q in problem]
        sol = [self.sol_word2int[q] if q in self.sol_word2int else self.sol_word2int["<unk>"] for q in sol]
        return ((problem,sol))


class GloveEmbeddings():
    def __init__(self, embed_dim, word2idx):
        self.embed_dim = embed_dim
        self.word2idx = word2idx
        self.vocab_size = len(word2idx)
    
    def get_embedding_matrix(self):
        glove = GloVe(name='6B', dim=self.embed_dim)
        embedding_matrix = torch.zeros((self.vocab_size, self.embed_dim)) #init zeros
        
        for i in pos_special_tokens_prob[:3]:
            embedding_matrix[i] = torch.randn(self.embed_dim)
            
        for k, v in self.word2idx.items():        
            if k in glove.stoi:
                embedding_matrix[v] = torch.tensor(glove.vectors[glove.stoi[k]])
            elif k not in special_tags:
                embedding_matrix[v] = embedding_matrix[pos_special_tokens_prob[2]]
#         embedding_matrix[]
        return embedding_matrix
    

class AttentionNetwork_b(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionNetwork_b, self).__init__()
        self.attention_fc = nn.Linear(hidden_dim * 3, hidden_dim)
        self.vector_projection = nn.Linear(hidden_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, last_hidden_state, encoder_outputs):
        # Correctly repeating the hidden state to match the batch size of the encoder outputs
        repeated_hidden = last_hidden_state.repeat(encoder_outputs.size(1), 1, 1).transpose(0, 1)
        energy = torch.cat((repeated_hidden, encoder_outputs), dim=2)
        energy = torch.tanh(self.attention_fc(energy))
        attention_scores = self.vector_projection(energy).squeeze(2)
        attention_weights = self.softmax(attention_scores)
        context_vector = attention_weights.unsqueeze(1).bmm(encoder_outputs)
        return context_vector, attention_weights
    

    
          
class LSTMEncoder_b(nn.Module):
    def __init__(self, embedding_dim, hidden_units=256, embed_matrix=None):
        super(LSTMEncoder_b, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.embedding = nn.Embedding.from_pretrained(embed_matrix, padding_idx=pos_special_tokens_prob[3])
        self.dropout = nn.Dropout(0.5) 
        self.lstm = nn.LSTM(embedding_dim, hidden_units, num_layers=1, batch_first=True, 
                            dropout=0.5, bidirectional=True)
        self.hidden_layer = nn.Linear(hidden_units * 2, hidden_units)  
        self.cell_layer = nn.Linear(hidden_units * 2, hidden_units)

    def forward(self, inputs):
        embedded_inputs = self.dropout(self.embedding(inputs))  
        lstm_out, (hidden, cell) = self.lstm(embedded_inputs)
        hidden = self.hidden_layer(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.cell_layer(torch.cat((cell[0:1], cell[1:2]), dim=2))
        return lstm_out, (hidden, cell)


 
    
class LSTMAttnDecoder_b(nn.Module):
    def __init__(self, vocab_size, embedding_dim, units=256):
        super(LSTMAttnDecoder_b, self).__init__()
        self.embed_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=pos_special_tokens_sol[3])  
        self.dropout_layer = nn.Dropout(0.5)
        self.attn_layer = AttentionNetwork_b(units)
        self.lstm = nn.LSTM(embedding_dim + units * 2, units, num_layers = 1, bidirectional = False, dropout=0.5, batch_first=True)
        self.out = nn.Linear(units, vocab_size)

    def forward(self, input_seq, hidden_state, encoder_outputs):
        embedded = self.dropout_layer(self.embed_layer(input_seq)).unsqueeze(1)
        attn_context, attn_weights = self.attn_layer(hidden_state[0], encoder_outputs)
        lstm_input = torch.cat([embedded, attn_context], dim=2)
        lstm_out, (h_n, c_n) = self.lstm(lstm_input, hidden_state)
        output = self.out(lstm_out)
        return output, (h_n, c_n)

class Seq2Seq_b(nn.Module):
    def __init__(self, embedding=None,embedding_dim=200,max_len_pred=None):
        super(Seq2Seq_b, self).__init__()
        self.lstm_encoder=LSTMEncoder_b(embedding_dim=embedding_dim,embed_matrix=embedding)
        self.lstm_decoder=LSTMAttnDecoder_b(vocab_size=max_len_pred,embedding_dim=embedding_dim)
        self.decoder_vocab_size=max_len_pred
    def forward(self, source_seq, target_seq, tf=0.9):
        src_batch_size = source_seq.size(0)
        tgt_seq_len = target_seq.size(1)
        
        encoder_states, (encoder_hidden, encoder_cell) = self.lstm_encoder(source_seq)

        decoder_vocab_size = self.decoder_vocab_size
        decoder_outputs = torch.zeros(src_batch_size, tgt_seq_len, decoder_vocab_size)
        predicted_sequence = torch.zeros(src_batch_size, tgt_seq_len)

        decoder_input_token = target_seq[:, 0]  # Initial decoder input
        predicted_sequence[:, 0] = decoder_input_token
        
        for step in range(1, tgt_seq_len):
            decoder_output, (encoder_hidden, encoder_cell) = self.lstm_decoder(decoder_input_token, (encoder_hidden, encoder_cell),encoder_states)
            decoder_output = decoder_output.squeeze(1)
            decoder_outputs[:, step, :] = decoder_output
            use_teacher_forcing = np.random.random() < tf
            decoder_input_token = target_seq[:, step] if use_teacher_forcing else decoder_output.argmax(dim=1)
            predicted_sequence[:, step] = decoder_output.argmax(dim=1)

        return decoder_outputs, predicted_sequence


      
class lstm_encoder_a(nn.Module):
    def __init__(self, embedding_dim, hidden_units=512, embed_matrix=None,padding_idx=3):
        super(lstm_encoder_a, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.embedding = nn.Embedding.from_pretrained(embed_matrix, padding_idx=padding_idx)
        self.dropout = nn.Dropout(0.5)  # Reintroduced dropout layer
        self.lstm = nn.LSTM(embedding_dim, hidden_units, num_layers=1, batch_first=True, 
                            dropout=0.5, bidirectional=True)
        self.hidden_layer = nn.Linear(hidden_units * 2, hidden_units) 
        self.cell_layer = nn.Linear(hidden_units * 2, hidden_units)

    def forward(self, inputs):
        embedded_inputs = self.dropout(self.embedding(inputs)) 
        lstm_out, (hidden, cell) = self.lstm(embedded_inputs)
        hidden = self.hidden_layer(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.cell_layer(torch.cat((cell[0:1], cell[1:2]), dim=2))
        return lstm_out, (hidden, cell)



class lstm_decoder_a(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units=512,padding_idx=18):
        super(lstm_decoder_a, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(0.5)  
        self.lstm = nn.LSTM(embedding_dim, hidden_units, num_layers=1, batch_first=True, dropout=0.5)
        self.output_layer = nn.Linear(hidden_units, vocab_size)

    def forward(self, inputs, initial_state):
        embedded_inputs = self.dropout(self.embedding(inputs)).unsqueeze(1)
        lstm_out, (hidden, cell) = self.lstm(embedded_inputs, initial_state)
        output = self.output_layer(lstm_out.squeeze(1))
        return output, (hidden, cell)
    


class Seq2Seq_a(nn.Module):
    def __init__(self,embedding=None,embedding_dim=100,max_len_pred=200):
        super(Seq2Seq_a, self).__init__()
        self.encoder = lstm_encoder_a(embedding_dim=embedding_dim, embed_matrix=embedding)
        self.decoder = lstm_decoder_a(vocab_size=max_len_pred,embedding_dim=embedding_dim)

    def forward(self, source_seq, target_seq, tf=0.9):
        src_batch_size = source_seq.size(0)
        tgt_seq_len = target_seq.size(1)
        
        encoder_states, (encoder_hidden, encoder_cell) = self.encoder(source_seq)

        decoder_vocab_size = self.decoder.vocab_size
        decoder_outputs = torch.zeros(src_batch_size, tgt_seq_len, decoder_vocab_size)
        predicted_sequence = torch.zeros(src_batch_size, tgt_seq_len)

        decoder_input_token = target_seq[:, 0]  # Initial decoder input
        predicted_sequence[:, 0] = decoder_input_token
        
        for step in range(1, tgt_seq_len):
            decoder_output, (encoder_hidden, encoder_cell) = self.decoder(decoder_input_token, (encoder_hidden, encoder_cell))
            decoder_output = decoder_output.squeeze(1)
            decoder_outputs[:, step, :] = decoder_output
            use_teacher_forcing = np.random.random() < tf
            decoder_input_token = target_seq[:, step] if use_teacher_forcing else decoder_output.argmax(dim=1)
            predicted_sequence[:, step] = decoder_output.argmax(dim=1)

        return decoder_outputs, predicted_sequence


class BeamSearch3():
    def __init__(self, model,single_data,model_type, device,  max_target_len=80, beam_size=beam_size):
        self.model = model
        self.device = device
        self.start_token = pos_special_tokens_sol[0]
        self.en_ht = None
        self.en_ct = None
        self.encoder_out=None
        self.decoder_input_token=None
        self.max_target_len = max_target_len
        self.beam_size = beam_size
        self.single_data=single_data
        self.get_encoder_outputs(model_type)
        self.model_type=model_type
        self.beam = [([self.start_token], (self.en_ht, self.en_ct), 0)]


    def get_encoder_outputs(self,model_type):
        if(model_type==1):
            x=self.single_data[0].to(device)
            self.encoder_out, (self.en_ht, self.en_ct) = self.model.encoder(x)
        
        elif(model_type==2 ):
            x=self.single_data[0].to(device)
            self.encoder_out, (self.en_ht, self.en_ct) = self.model.lstm_encoder(x)

        elif(model_type==3 or model_type==4):
            src=self.single_data[0].to(device)
            src_mask=self.single_data[2].to(device)
            batch_size = src.shape[0]
            self.encoder_out = self.model.module.encoder(src, src_mask)
            self.en_ht = torch.zeros(1, batch_size, self.model.module.decoder.rnn.hidden_size).to(self.device)
            self.en_ct = torch.zeros(1, batch_size, self.model.module.decoder.rnn.hidden_size).to(self.device)
            

    def search(self):
        for _ in range(self.max_target_len - 1):
            self._expand_beam(self.model_type)
            self.beam.sort(key=lambda x: x[2])
            self.beam = self.beam[:self.beam_size]

        best_candidate = self.beam[0][0]
        decoded_words = self._construct_output(best_candidate)
        return decoded_words

    def _expand_beam(self,model_type):
        new_beam = []
        for sequence, (ht, ct), score in self.beam:
            prev_token = torch.LongTensor([sequence[-1]]).to(self.device)

            if(model_type==1):
                decoder_out, (ht, ct) = self.model.decoder(prev_token, (ht, ct))
            elif(model_type==2):
                decoder_out, (ht, ct) = self.model.lstm_decoder(prev_token, (ht, ct),self.encoder_out)
            elif(model_type==3 or model_type==4):
                decoder_out, ht, ct= self.model.module.decoder(prev_token, ht, ct,self.encoder_out)

            decoder_out = decoder_out.squeeze(1)
            top_vals, top_inds = decoder_out.topk(self.beam_size, dim=1)

            self._add_candidates(new_beam, sequence, ht, ct, score, top_vals, top_inds)

        self.beam = new_beam

    def _add_candidates(self, new_beam, sequence, ht, ct, score, top_vals, top_inds):
        for j in range(self.beam_size):
            new_word_idx = top_inds[0][j]
            new_seq = sequence + [new_word_idx.item()]
            new_word_prob = torch.log(top_vals[0][j])
            updated_score = score - new_word_prob
            new_candidate = (new_seq, (ht, ct), updated_score)
            new_beam.append(new_candidate)

    def _construct_output(self, best_candidate):
        decoded_words = torch.zeros(1, self.max_target_len)
        for t, idx in enumerate(best_candidate):
            decoded_words[:, t] = torch.LongTensor([idx])
        return decoded_words
    


def generate_csv(model,data,loader):
    model.eval()
    prediction=[]
    for batch in loader:
        x=batch[0].to(device)
        y=batch[1].to(device)
        o,w=model(x,y,tf=0)
        prediction.append(w)
    predicted_data_strings=[]
    for batch in prediction:
        for one_data in batch:
            one_data=one_data.to(torch.int)
            one_data_in_string=""
            conv_int_word=[]
            for pos,value in enumerate(one_data):
                conv_int_word.append(data.sol_int2word[value.item()])
            predicted_data_strings.append(conv_int_word)
    return predicted_data_strings

def extract_sol(data2str):
    complete_data=[]
    for each_data in data2str:
        s=""
        flag=0
        for item in each_data[1:]:
            if(item=="<eos>" or item=="<pad>"):
                complete_data.append(s)
                flag=1
                break
            else:
                s+=str(item)
        if(flag==0):
            complete_data.append(s)
    return complete_data


def generate_sol(int2data,name,data_name):
    json_data=[]
    for k,s in zip(data_name.data,int2data):
        d={
            "Problem":k[0],
            "predicted":s,
        }
        json_data.append(d)
    with open(name, 'w') as json_file:
        json.dump(json_data, json_file,indent=4)
    print("prediction json generated!!")

def convtostr(prediction,data):
    predicted_data_strings=[]
    for batch in prediction:
        for one_data in batch:
            one_data=one_data.to(torch.int)
            one_data_in_string=""
            conv_int_word=[]
            for pos,value in enumerate(one_data):
                conv_int_word.append(data.sol_int2word[value.item()])
            predicted_data_strings.append(conv_int_word)
    return predicted_data_strings

def get_me_final_file(op_test,data,file_name):
    prediction_in_str=convtostr(op_test,data)
    good=extract_sol(prediction_in_str)
    generate_sol(good,file_name,data)


def fun1(model_file,beam_size,model_type,test_data_file):
    train_path="train.json"
    test_path=test_data_file
    modify_and_overwrite_json(test_data_file)
    test_loader=None
    test_data=None
    model=None
    trained_model=None
    mt=-1
    

    if(model_type=="lstm_lstm"):
        globals()['lstm_encoder'] = lstm_encoder_a
        globals()['lstm_decoder'] = lstm_decoder_a
        globals()['Seq2Seq'] = Seq2Seq_a
        trained_model = torch.load(model_file)
        
        train_data=load_data_train_a_b(train_path)
        test_data=load_data_test_a_b(test_path,train_data)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_a_b)
        glove=GloveEmbeddings(200,test_data.problem_word2int)
        embedding=glove.get_embedding_matrix()
        model=Seq2Seq_a(embedding=embedding,embedding_dim=200,max_len_pred=len(train_data.sol_unique_words))
        mt=1
    elif(model_type=="lstm_lstm_attn"):
        globals()['LSTMEncoder'] = LSTMEncoder_b
        globals()['AttentionNetwork']=AttentionNetwork_b
        globals()['LSTMAttnDecoder'] = LSTMAttnDecoder_b
        globals()['Seq2Seq'] = Seq2Seq_b
        trained_model = torch.load(model_file)
        
        train_data=load_data_train_a_b(train_path)
        test_data=load_data_test_a_b(test_path,train_data)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_a_b)
        glove=GloveEmbeddings(200,test_data.problem_word2int)
        embedding=glove.get_embedding_matrix()
        model=Seq2Seq_b(embedding=embedding,embedding_dim=200,max_len_pred=len(train_data.sol_unique_words))
        mt=2

    elif(model_type=="bert_lstm_attn_frozen"):
        globals()['BERT_Encoder'] = BERT_Encoder_c
        globals()['Attention']=Attention_c
        globals()['LSTM_Decoder'] = LSTM_Decoder_c
        globals()['Seq2Seq'] = Seq2Seq_c
        trained_model = torch.load(model_file)
        
        train_data=load_data_train_c_d(train_path)
        test_data=load_data_test_c_d(test_path,train_data)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_c_d)
        model=Seq2Seq_c(max_len_pred=len(train_data.sol_unique_words),device=device)
        mt=3
        
    elif(model_type=="bert_lstm_attn_tuned"):
        globals()['BERT_Encoder'] = BERT_Encoder_d
        globals()['Attention']=Attention_d
        globals()['LSTM_Decoder'] = LSTM_Decoder_d
        globals()['Seq2Seq'] = Seq2Seq_d
        trained_model = torch.load(model_file)
        
        train_data=load_data_train_c_d(train_path)
        test_data=load_data_test_c_d(test_path,train_data)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_c_d)
        model=Seq2Seq_d(max_len_pred=len(train_data.sol_unique_words),device=device)
        mt=4
    
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.load_state_dict(trained_model.state_dict())
    
    
    op_test=[]
    for i,one_data in enumerate(test_loader):
        beam=BeamSearch3(model=model,single_data=one_data,model_type=mt, device=device,  max_target_len=300, beam_size=beam_size)
        op_test.append(beam.search())
        print(f"{i} sentence completed")
    get_me_final_file(op_test,test_data,test_data_file)

   

def main():
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--model_file', type=str, help='Path to the trained model')
    parser.add_argument('--beam_size', type=int, choices=[1, 10, 20], help='Beam size')
    parser.add_argument('--model_type', type=str, choices=['lstm_lstm', 'lstm_lstm_attn', 'bert_lstm_attn_frozen', 'bert_lstm_attn_tuned'], help='Model type')
    parser.add_argument('--test_data_file', type=str, help='Path to the JSON file containing the problems')

    args = parser.parse_args()

    model_file = args.model_file
    beam_size = args.beam_size
    model_type = args.model_type
    test_data_file = args.test_data_file


    fun1(model_file,beam_size,model_type,test_data_file)

if __name__ == "__main__":
    main()
