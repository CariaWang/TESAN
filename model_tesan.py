import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F


# module for self-attention
class SAN(nn.Module):
    def __init__(self, in_dim, h_dim):
        super(SAN, self).__init__()
        self.h_dim = h_dim
        self.Wx = nn.Linear(in_dim, h_dim)
        self.Wx_drop = nn.Dropout(0.2)
        self.Wa = nn.Linear(h_dim, h_dim)
        self.Ua = nn.Linear(h_dim, h_dim)
        self.drop = nn.Dropout(0.2)
        # initial
        nn.init.xavier_normal(self.Wx.weight, 1)
        nn.init.xavier_normal(self.Wa.weight, 1)
        nn.init.xavier_normal(self.Ua.weight, 1)
        nn.init.constant(self.Wx.bias, 0.0)
        nn.init.constant(self.Wa.bias, 0.0)
        nn.init.constant(self.Ua.bias, 0.0)

    def forward(self, word_vecs, mask):
        h_vecs = F.tanh(self.Wx(word_vecs))   # shape=[B, L, H]
        h_vecs = self.Wx_drop(h_vecs)
        temp_a = self.Wa(h_vecs).unsqueeze(2) # shape=[B, L, 1, H]
        temp_b = self.Ua(h_vecs).unsqueeze(1) # shape=[B, 1, L, H]
        temp = F.elu(temp_a + temp_b)         # [B, L, L, H]
        a = F.softmax(temp + mask.unsqueeze(1), dim=2)
        c_vecs = torch.sum(torch.mul(a, h_vecs.unsqueeze(1).expand(h_vecs.shape[0],h_vecs.shape[1],h_vecs.shape[1],h_vecs.shape[2])), dim=2) # [B, L, H]
        c_vecs = self.drop(c_vecs)
        return c_vecs


# module for topic-enhanced self-attention
class TESA(nn.Module):
    def __init__(self, in_dim, h_dim):
        super(TESA, self).__init__()
        # pos embedding
        self.pos_emb = nn.Embedding(20, in_dim)
        # context attention
        self.x_drop = nn.Dropout(0.2)
        self.san = SAN(in_dim, h_dim)
        # global attention
        self.Wc = nn.Linear(h_dim, h_dim)
        self.V = nn.Linear(h_dim, h_dim)
        self.V_drop = nn.Dropout(0.2)
        self.drop = nn.Dropout(0.2)
        # initial
        nn.init.xavier_normal(self.pos_emb.weight, 1)
        nn.init.xavier_normal(self.Wc.weight, 1)
        nn.init.xavier_normal(self.V.weight, 1)
        nn.init.constant(self.Wc.bias, 0.0)
        nn.init.constant(self.V.bias, 0.0)

    def forward(self, word_vecs, dt_vec, mask):
        # position embedding
        pos_idx = Variable(torch.arange(word_vecs.shape[1]).long()).cuda()
        pos_vecs = self.pos_emb(pos_idx)

        # self attention
        input_vecs = word_vecs + pos_vecs
        input_vecs = self.x_drop(input_vecs)
        con_vecs = self.san(input_vecs, mask)

        # topical attention
        temp = F.tanh(self.Wc(con_vecs))
        dt_V = self.V_drop(F.tanh(self.V(dt_vec)))
        a = F.softmax(torch.matmul(temp, dt_V.unsqueeze(2))+mask, dim=1)
        doc_vec = torch.sum(torch.mul(a, con_vecs), dim=1)
        doc_vec = self.drop(doc_vec)

        return doc_vec


# module for neural topic model
class NTM(nn.Module):
    def __init__(self, net_arch):
        super(NTM, self).__init__()
        ac = net_arch
        self.net_arch = net_arch
        # encoder
        self.en1_fc     = nn.Linear(ac.num_input, ac.en1_units)
        self.en2_fc     = nn.Linear(ac.en1_units, ac.en2_units)
        self.en2_drop   = nn.Dropout(0.2)
        self.mean_fc    = nn.Linear(ac.en2_units, ac.num_topic)
        self.mean_bn    = nn.BatchNorm1d(ac.num_topic)               # bn for mean
        self.logvar_fc  = nn.Linear(ac.en2_units, ac.num_topic)
        self.logvar_bn  = nn.BatchNorm1d(ac.num_topic)               # bn for logvar
        # z
        self.p_drop     = nn.Dropout(0.2)
        # decoder
        self.topic_emb  = nn.Linear(ac.num_topic, ac.h_dim, bias=False)
        self.word_emb = nn.Linear(ac.h_dim, ac.num_input)
        self.decoder_bn = nn.BatchNorm1d(ac.num_input)           # bn for decoder

        # prior mean and variance as constant buffers
        prior_mean   = torch.Tensor(1, ac.num_topic).fill_(0)
        prior_var    = torch.Tensor(1, ac.num_topic).fill_(ac.variance)
        prior_logvar = prior_var.log()
        self.register_buffer('prior_mean',    prior_mean)
        self.register_buffer('prior_var',     prior_var)
        self.register_buffer('prior_logvar',  prior_logvar)

        # initial
        nn.init.xavier_normal(self.en1_fc.weight, 1)
        nn.init.xavier_normal(self.en2_fc.weight, 1)
        nn.init.xavier_normal(self.mean_fc.weight, 1)
        nn.init.xavier_normal(self.logvar_fc.weight, 1)
        nn.init.xavier_normal(self.topic_emb.weight, 1)
        nn.init.xavier_normal(self.word_emb.weight, 1)
        nn.init.constant(self.en1_fc.bias, 0.0)
        nn.init.constant(self.en2_fc.bias, 0.0)
        nn.init.constant(self.mean_fc.bias, 0.0)
        nn.init.constant(self.logvar_fc.bias, 0.0)
        nn.init.constant(self.word_emb.bias, 0.0)

    def forward(self, input, compute_loss, avg_loss):
        # compute posterior
        en1 = F.softplus(self.en1_fc(input))
        en2 = F.softplus(self.en2_fc(en1))
        en2 = self.en2_drop(en2)
        posterior_mean   = self.mean_bn(self.mean_fc  (en2))          # posterior mean
        posterior_logvar = self.logvar_bn(self.logvar_fc(en2))          # posterior log variance
        posterior_var    = posterior_logvar.exp()
        # take sample
        eps = Variable(input.data.new().resize_as_(posterior_mean.data).normal_()) # noise
        z = posterior_mean + posterior_var.sqrt() * eps                 # reparameterization
        p = F.softmax(z, dim=1)                                         # mixture probability
        p = self.p_drop(p)
        # use p for doing reconstruction
        dt_vec = F.tanh(self.topic_emb(p))
        recon = F.softmax(self.decoder_bn(self.word_emb(dt_vec)), dim=1)   # reconstructed distribution over vocabulary

        if compute_loss:
            return recon, self.loss(input, recon, posterior_mean, posterior_logvar, posterior_var, avg_loss), dt_vec
        else:
            return recon, dt_vec

    def loss(self, input, recon, posterior_mean, posterior_logvar, posterior_var, avg=True):
        # NL
        NL  = -(input * (recon+1e-10).log()).sum(1)
        # KLD, see Section 3.3 of Akash Srivastava and Charles Sutton, 2017, 
        # https://arxiv.org/pdf/1703.01488.pdf
        prior_mean   = Variable(self.prior_mean).expand_as(posterior_mean)
        prior_var    = Variable(self.prior_var).expand_as(posterior_mean)
        prior_logvar = Variable(self.prior_logvar).expand_as(posterior_mean)
        var_division    = posterior_var  / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        # put KLD together
        KLD = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - self.net_arch.num_topic)
        # loss
        loss = (NL + KLD)
        # in traiming mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss.mean()
        else:
            return loss

# the whole module
class TESAN(nn.Module):
    def __init__(self, net_arch):
        super(TESAN, self).__init__()
        self.ntm = NTM(net_arch)
        self.tesa = TESA(net_arch.in_dim, net_arch.h_dim)
        # fusion gate
        self.Wg = nn.Linear(net_arch.h_dim, net_arch.h_dim)
        self.Ug = nn.Linear(net_arch.h_dim, net_arch.h_dim)
        self.f_drop = nn.Dropout(0.2)
        # classifier
        self.classifier = nn.Linear(net_arch.h_dim, net_arch.num_class)

        # initial
        nn.init.xavier_normal(self.Wg.weight, 1)
        nn.init.xavier_normal(self.Ug.weight, 1)
        nn.init.xavier_normal(self.classifier.weight, 1)
        nn.init.constant(self.Wg.bias, 0.0)
        nn.init.constant(self.Ug.bias, 0.0)
        nn.init.constant(self.classifier.bias, 0.0)

    def forward(self, input, word_vecs, mask, compute_loss=False, avg_loss=True):
        # NTM
        recon, ntm_loss, dt_vec = self.ntm(input, compute_loss, avg_loss)

        # topic-enhanced self-attention
        doc_vec = self.tesa(word_vecs, dt_vec, mask)

        # fusion gate
        gate = F.sigmoid(self.Wg(doc_vec)+self.Ug(dt_vec))
        fea_vec = torch.mul(gate, doc_vec)+torch.mul(1.0-gate, dt_vec)
        fea_vec = self.f_drop(fea_vec)

        # classifier
        pre_vec = self.classifier(fea_vec)
        out = F.log_softmax(pre_vec, dim=1)

        if compute_loss:
            return recon, ntm_loss, out
        else:
            return recon, out