import copy

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, RobertaConfig, RobertaModel, AutoTokenizer, AutoModel
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder, ExtTgtTransformerEncoder
from models.optimizers import Optimizer
import numpy as np

def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))


    return optim

def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim

def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim

def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator


class Bert(nn.Module):
    def __init__(self, large, temp_dir, model, device, finetune=False):
        super(Bert, self).__init__()
        if(large):
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            if model == "robert":
                self.model = RobertaModel.from_pretrained('roberta-base', cache_dir=temp_dir)
            if model == "bert":
                self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
            if model == "pubmed":
                model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
                self.model = AutoModel.from_pretrained(model_name,cache_dir=temp_dir).to(device)
            if model == "biobert":
                model_name = 'dmis-lab/biobert-v1.1'
                self.model = AutoModel.from_pretrained(model_name,cache_dir=temp_dir).to(device)
        self.finetune = finetune

    def forward(self, x, segs, mask):
        if(self.finetune):
            output = self.model(input_ids=x, token_type_ids=segs, attention_mask=mask)
            top_vec = output.last_hidden_state
        else:
            self.eval()
            with torch.no_grad():
                output = self.model(input_ids=x, token_type_ids=segs, attention_mask=mask)
                top_vec = output.last_hidden_state
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.model, device, args.finetune_bert)
        self.drop_out = nn.Dropout(args.ext_dropout)
        self.layer_norm = nn.LayerNorm(args.topic_number, eps=1e-6)
        self.doc_layer_norm = nn.LayerNorm(args.voc_size, eps=1e-6)
        self.bi_grus = nn.GRU(input_size=self.bert.model.config.hidden_size, hidden_size=self.bert.model.config.hidden_size,num_layers=2, batch_first=True, bidirectional=True)
        self.bi_grus_doc = nn.GRU(input_size=args.topic_number,
                              hidden_size=args.topic_number,
                              num_layers=1, batch_first=True, bidirectional=True)
        self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads, args.ext_dropout, args.ext_layers)
        self.ext_tgt_layer = ExtTgtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,args.ext_dropout, args.topic_number, args.ext_layers)
        self.join = nn.Linear(2 * self.bert.model.config.hidden_size, self.bert.model.config.hidden_size)
        self.linear = nn.Linear(args.voc_size, args.topic_number)
        #self.attention = DotProductAttention(self.bert.model.config.hidden_size)
        self.topic_linear = nn.Linear(args.topic_number, args.topic_number)
        self.topic_sig_linear = nn.Linear(args.topic_number, args.topic_number)
        self.doc_linear = nn.Linear(args.topic_number, args.topic_number)
        self.classifier = Classifier(args.topic_number)
        self.decoder = nn.Linear(args.topic_number, args.voc_size)
        self.a = 1 * np.ones((1, args.topic_number))
        self.mu2 = torch.tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T).to(device)
        self.var2 = torch.tensor((((1.0 / self.a) * (1 - (2.0 / args.topic_number))).T + (1.0 / (args.topic_number * args.topic_number)) * np.sum(1.0 / self.a, 1)).T).to(device)
        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
                                     num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads, intermediate_size=args.ext_ff_size)
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device)

    def pooling(self, last_hidden_state):
        if self.args.pooling == "none":
            node_feature = pooler_output
        elif self.args.pooling == "mean":
            node_feature = torch.mean(last_hidden_state, dim=1)
        elif self.args.pooling == "max":
            node_feature = torch.max(last_hidden_state, dim=1)[0]
        else:
            mean_feat = torch.mean(last_hidden_state, dim=1)
            max_feat = torch.max(last_hidden_state, dim=1)[0]
            feat = torch.cat([mean_feat, max_feat], 1)
            #print (feat.shape, self.join)
            node_feature = self.join(feat)

        return node_feature.unsqueeze(1)

    def forward(self, src, segs, clss, mask_src, mask_cls, src_idx, data_feature,labels,flag_train="test"):
        #sents_mean
        top_vec_mean = self.bert(src, segs, mask_src)
        top_vecs_mean = torch.split(top_vec_mean, src_idx.tolist())
        sents_vec = [vec.reshape(vec.size(0) * vec.size(1), vec.size(-1))[clss[i], :] for i, vec in enumerate(top_vecs_mean)]
        sents_vec = torch.stack(sents_vec)
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sents_vec = self.ext_tgt_layer(sents_vec, mask_cls)
        sents_vec = sents_vec * mask_cls[:, :, None].float()

        sents_vec_mean = self.topic_linear(sents_vec)
        sents_vec_mean = self.layer_norm(sents_vec_mean)
        sents_vec_mean = sents_vec_mean * mask_cls[:, :, None].float()

        sents_vec_sig = self.topic_linear(sents_vec)
        sents_vec_sig = self.layer_norm(sents_vec_sig)
        sents_vec_sig = sents_vec_sig * mask_cls[:, :, None].float()

        #pooling
        doc_input = torch.mean(sents_vec,1)
        #print(sents_vec_mean.shape, sents_vec_sig.shape, doc_input.shape)
        #doc_input = self.attention(sent_all, sents_vec)
        doc_input = self.doc_linear(doc_input)
        #doc_input = self.layer_norm(doc_input)

        #print(doc_input.shape)
        #noise
        eps = torch.normal(0,1,size=(sents_vec_mean.size(0), sents_vec_mean.size(1))).to(self.device)
        sents_vec_sig = torch.exp(sents_vec_sig)
        sents_embed = torch.add(sents_vec_mean, torch.mul(torch.sqrt(sents_vec_sig),torch.unsqueeze(eps,-1)))*mask_cls[:, :, None].float()
        sent_scores = self.classifier(sents_embed, mask_cls)

        ## doc encoder
        #if flag_train == "train":
        #    sent_to_doc = sents_embed * labels.unsqueeze(dim=-1)
        #else:
        #    sent_to_doc = sents_embed * sent_scores.unsqueeze(dim=-1)
        #data_feature_input = torch.cat((data_feature,sent_to_doc.mean(1)),-1)
        doc_mean = self.drop_out(F.softplus(self.linear(data_feature)+doc_input))
        doc_mean = self.layer_norm(doc_mean)
        doc_sig = self.drop_out(F.softplus(self.linear(data_feature)+doc_input))
        doc_sig = self.layer_norm(doc_sig)
        sig = torch.exp(doc_sig)
        eps_doc = torch.normal(0,1, size=(doc_mean.size(0), doc_mean.size(1))).to(self.device)
        #print(doc_mean.shape, doc_sig.shape, eps_doc.shape)
        doc_embed = F.softmax(torch.add(doc_mean, torch.mul(torch.sqrt(sig),eps_doc)))
        #print(doc_embed.shape)
        ## decoder
        recon = F.softmax(self.doc_layer_norm(self.decoder(doc_embed)))

        #cos similarity
        similarity = torch.nn.CosineSimilarity(dim=-1)(sents_embed, doc_embed.unsqueeze(1).repeat(1, sents_embed.size()[1], 1))
        rec_loss = -(data_feature * (recon + 1e-10).log()).sum(1)
        latent_loss = (0.5 / self.args.topic_number) * (torch.sum(torch.div(sig, self.var2), 1) + \
                    torch.sum(torch.multiply(torch.div((self.mu2 - doc_mean), self.var2),
                                (self.mu2 - doc_mean)),1) - self.args.topic_number + \
                                 torch.sum(torch.log(self.var2), 1) - torch.sum(doc_sig, 1))
        topic_loss = (rec_loss + latent_loss).mean()
        return sent_scores, mask_cls, similarity, topic_loss, doc_embed, sents_embed


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(
                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
                                     num_hidden_layers=args.enc_layers, num_attention_heads=8,
                                     intermediate_size=args.enc_ff_size,
                                     hidden_dropout_prob=args.enc_dropout,
                                     attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if(args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, None
