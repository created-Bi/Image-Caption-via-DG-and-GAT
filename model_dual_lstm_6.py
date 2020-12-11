import torch
from torch import nn
import torchvision
from torch.nn.utils.rnn import pack_padded_sequence
from utils import selfAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1st, we need a image feature extractor. Initial: out_image_size, Input: image, Output: image feature
# output: batch_size * out_image_size, out_image_size, feature_num(ResNet-101:2048)
class ImageExtractor(nn.Module):
    def __init__(self, out_image_size=14):
        super(ImageExtractor, self).__init__()
        self.out_image_size = out_image_size

        resnet = torchvision.models.resnet101(pretrained=True).to(device)

        # Remove the linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((out_image_size, out_image_size))
        self.adaptive_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fine_tune()

    def forward(self, images):
        """
        Feed Forward propagation
        :param images [ batch_size, 3, image_size, image_size(256)]
        :return: image feature [ batch_size, out_image_size, out_image_size, 2048 ]
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out_root = self.adaptive_max_pool(out.contiguous()).squeeze()  # batch_size, 2048
        out_caption = self.adaptive_avg_pool(out.contiguous()).permute(0, 2, 3, 1)  # batch_size, 14, 14, 2048
        # print('image processing...')
        return out_root, out_caption

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the extractor.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune the last convolutional block
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    def __init__(self, input1_dim, input2_dim, hidden_dim):
        super(Attention, self).__init__()
        self.Linear1 = nn.Linear(input1_dim, hidden_dim)  # batch_size, attention_dim
        self.Linear2 = nn.Linear(input2_dim, hidden_dim)  # batch_size, iter_times, attention_dim
        self.full_att2_1 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input1, input2):
        enc_input1 = self.Linear1(input1)  # batch_size, attention_dim
        enc_input2 = self.Linear2(input2)  # batch_size, 196, attention_dim / batch_size, iter_times, attention_dim
        attention_weight = self.full_att2_1(self.relu(torch.add(enc_input2, enc_input1.unsqueeze(1)))).squeeze(2)
        # attention_weight and alpha: batch_size, 196
        alpha = self.softmax(attention_weight)
        att_input2 = (input2 * alpha.unsqueeze(2)).sum(dim=1)  # batch_size, image_feature_dim / encoder_embed_dim
        return att_input2, alpha


class RootClassifier(nn.Module):
    def __init__(self, vocab_size, word_num=1, encoder_dim=2048):
        super(RootClassifier, self).__init__()
        self.word_num = word_num
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim // 2),
            nn.ReLU(),
            nn.Linear(encoder_dim // 2, vocab_size)
        )

    def forward(self, encoder_out):
        output = self.classifier(encoder_out)  # batch_size, vocab_size
        _, labels_idx = output.topk(self.word_num, 1, True, True)
        return labels_idx  # batch_size, word_num


class CaptionGeneratin(nn.Module):
    def __init__(self, vocab_size, triple_size, dis_size, lstm_hidden_dim, attention_dim, embedd_dim, encoder_dim=2048,
                 dropout=0.5):
        '''
        :param encoder_hidden_dim:  (num_words + 1) * embedd_dim
        :param lstm_hidden_dim:  512
        :param embedd_dim: 512
        :param hidden_dim: 512
        :param vocab_size: 7464
        :param cap_embedding:
        :param dropout: 0.5
        '''
        super(CaptionGeneratin, self).__init__()
        self.vocab_size = vocab_size
        self.triple_size = triple_size
        self.dis_size = dis_size
        self.hidden_dim = lstm_hidden_dim
        self.encoder_dim = encoder_dim
        self.embedd_dim = embedd_dim

        # Only for the attention of encoder_feature's input order
        self.image_attention = Attention(lstm_hidden_dim, encoder_dim, attention_dim)
        self.image_beta = nn.Linear(lstm_hidden_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.graph_weight = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.Softmax(dim=1)
        )

        # Only for Generation
        self.triples_embedding = nn.Embedding(triple_size, embedd_dim)
        self.decode_step = nn.LSTMCell(embedd_dim + lstm_hidden_dim + encoder_dim, lstm_hidden_dim)
        self.decode_ds_step = nn.LSTMCell(embedd_dim + embedd_dim, lstm_hidden_dim)
        self.init_h1_cap = nn.Linear(encoder_dim,
                                     lstm_hidden_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c1_cap = nn.Linear(encoder_dim, lstm_hidden_dim)
        self.init_h2_cap = nn.Linear(embedd_dim,
                                     lstm_hidden_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c2_cap = nn.Linear(embedd_dim, lstm_hidden_dim)
        self.fc_trisize = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(lstm_hidden_dim, triple_size)
        )
        self.fc_vocab = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(lstm_hidden_dim, vocab_size)
        )
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.image_beta.bias.data.fill_(0)
        self.image_beta.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, all_image_feature, related_words_embeddings):
        '''
        :param all_image_feature: batch_size, num_pixels, image_feature_dim
        :param encoder_feature: # batch_size, iter_times, encoder_feature_dim
        :param image_feature: # batch_size, iter_times, image_feature_dim
        :return:  h_ds, c_ds, h_cap, c_cap
        '''
        all_image_feature = all_image_feature.mean(dim=1)
        related_words_embeddings = related_words_embeddings.mean(dim=1)
        h2 = self.init_h1_cap(all_image_feature)  # (batch_size, lstm_hidden_dim)
        c2 = self.init_c1_cap(all_image_feature)  # (batch_size, lstm_hidden_dim)
        h1 = self.init_h2_cap(related_words_embeddings)  # (batch_size, lstm_hidden_dim)
        c1 = self.init_c2_cap(related_words_embeddings)  # (batch_size, lstm_hidden_dim)
        return h1, c1, h2, c2

    def forward(self, encoder_out, encoded_triples, decode_lengths, captions_embedding,
                related_words_embeddings):  # , relations_embeddings
        '''
        :param encoder_out:  batch_size, 14, 14, encoder_dim
        :param decode_lengths: decode_lengths
        :param relations_embeddings:  batch_size, max_seq_length, embedding_dim
        :param captions_embedding:  batch_size, max_seq_length, embedding_dim
        :param related_words_embeddings: batch_size, iter_times, embedding_dim
        :return:
        '''
        batch_size = encoder_out.size(0)
        encoder_out = encoder_out.view(batch_size, -1, self.encoder_dim)  # batch_size, num_pixels, encoder_dim
        num_pixels = encoder_out.size(1)
        vocab_size, triple_size = self.vocab_size, self.triple_size
        # Initialization
        h1, c1, h2, c2 = self.init_hidden_state(encoder_out, related_words_embeddings)  # batch_size, lstm_hidden_dim
        triples_embedding = self.triples_embedding(encoded_triples).to(device)
        # new
        ds_predictions = torch.zeros(batch_size, max(decode_lengths), triple_size).to(device)
        caption_predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        end_score = torch.zeros(batch_size, max(decode_lengths)).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        for t in range(max(decode_lengths)):  # (0-25)
            batch_size_t = sum([l > t for l in decode_lengths])  # ( 0-32 )

            awe, alpha = self.image_attention(h2[:batch_size_t], encoder_out[:batch_size_t])
            gate_enc = self.sigmoid(self.image_beta(h2[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            awe = gate_enc * awe

            ''' if we should use a Linear before getting rws'''
            score = self.graph_weight(
                torch.bmm(related_words_embeddings[:batch_size_t], h1[:batch_size_t].unsqueeze(-1)).squeeze(
                    -1))  # batch_size, iter_times
            rws = (related_words_embeddings[:batch_size_t] * score.unsqueeze(-1)).sum(
                dim=1)  # batch_size, embedding_dim

            h1, c1 = self.decode_ds_step(torch.cat([triples_embedding[:batch_size_t, t, :], rws], dim=1),
                                         (h1[:batch_size_t], c1[:batch_size_t]))

            h2, c2 = self.decode_step(torch.cat([captions_embedding[:batch_size_t, t, :], h1, awe], dim=1),
                                      (h2[:batch_size_t], c2[:batch_size_t]))

            triple_preds = self.fc_trisize(h1)
            preds = self.fc_vocab(h2)
            ds_predictions[:batch_size_t, t, :] = triple_preds
            caption_predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
            end_score[:batch_size_t, t] = torch.exp(preds[:, 7463]) + torch.exp(triple_preds[:, 5282])
        return caption_predictions, ds_predictions, end_score, alphas


class Graph_Encoding(nn.Module):
    def __init__(self, vocab_size, dis_size, encoder_dim, embedd_dim, dropout, hidden_size, iter_times,
                 adjacent_words_embedding):
        super(Graph_Encoding, self).__init__()
        self.iter_times = iter_times
        self.embedd_dim = embedd_dim
        self.vocab_size = vocab_size
        self.dis_size = dis_size
        self.adjacent_words_embedding = adjacent_words_embedding

        self.IMGdedimension = nn.Linear(encoder_dim, encoder_dim)
        self.ROOTupdimension = nn.Linear(embedd_dim, embedd_dim)
        self.softmax = nn.Softmax(dim=1)
        self.relatedWordsClassifier = nn.Sequential(
            nn.Linear(encoder_dim + embedd_dim * 2, embedd_dim * 2),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embedd_dim * 2, vocab_size)
        )
        self.distance_2_1 = nn.Sequential(
            nn.Linear(encoder_dim + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, dis_size),
        )
        self.init_weights()

    def init_weights(self):
        self.IMGdedimension.bias.data.fill_(0)
        self.IMGdedimension.weight.data.uniform_(-0.1, 0.1)
        self.ROOTupdimension.bias.data.fill_(0)
        self.ROOTupdimension.weight.data.uniform_(-0.1, 0.1)

    def initialize_graph(self, root_node_idx, root_node_idxs, new_wprob_start_from_ck):
        root_node_idxs[:, 0] = root_node_idx
        new_wprob_start_from_ck[:, 0, :] = self.adjacent_words_embedding(root_node_idx)

    def forward(self, root_node_idx, mlp_out, words_embeddings, dis_embeddings):
        '''
                    1. use root to capture the core words
                    :param root_node_idx: batch_size,
                    :param mlp_out: batch_size, encoder_dim
                    :return distance_embedding_from_ck: batch_size, iter_times, embedding_dim
                    :return new_wprob_start_from_ck: batch_size, iter_times, vocab_size
                    :return new_rprob_start_from_ck: batch_size, iter_times, relations_size
                '''
        # print(root_node_idx)
        batch_size = root_node_idx.size(0)
        root_node_idxs = torch.LongTensor(batch_size * [self.iter_times * [0]]).to(device)
        distance_embedding_from_ck = torch.zeros(batch_size, self.iter_times, self.embedd_dim).to(device)
        distance_embedding_from_ck[:, 0, :] = dis_embeddings(
            torch.LongTensor(batch_size * [10]).to(device))  # batch_size, embedding_dim
        new_wprob_start_from_ck = torch.zeros(batch_size, self.iter_times, self.vocab_size).to(device)

        # print('initialize_graph ing....')
        self.initialize_graph(root_node_idx, root_node_idxs, new_wprob_start_from_ck)
        for i in range(self.iter_times - 1):
            # print('Iteration {}'.format(i))
            root_nodes_embeddings = words_embeddings(root_node_idxs[:, i])
            # print('root_nodes_embeddings: ',root_nodes_embeddings.size())
            root_input = torch.cat([self.ROOTupdimension(root_nodes_embeddings), self.IMGdedimension(mlp_out)],
                                   dim=1)  # batch_size, 1024 + 512 = 1536
            # print('root_input: ',root_input.size())
            distance = self.distance_2_1(root_input)  # batch_size, dis_size
            # print('distance1: ',distance.size())
            _, distance_idx = distance.topk(1, 1, True, True)
            # print('distance2: ',distance.size())
            distance_embedding = dis_embeddings(distance_idx.squeeze(1))  # batch_size, embedding_dim
            # print('distance_embedding: ',distance_embedding.size())
            related_input = torch.cat([root_input, distance_embedding], dim=1).unsqueeze(1).unsqueeze(1)
            # print('related_input: ',related_input.size())
            p_related_words = self.relatedWordsClassifier(
                related_input).squeeze()  # batch_size, vocab_size  (logSoftmax)
            # print('p_related_words: ',p_related_words.size())
            prob_from_gloabl_dt_graph = new_wprob_start_from_ck[:, i, :]  # batch_size, vocab_size
            flag = 1e-5 * (prob_from_gloabl_dt_graph < 0.0001) + 1.0 * (prob_from_gloabl_dt_graph >= 0.0001)
            if p_related_words.size(0) == 7464: p_related_words = p_related_words.unsqueeze(0)
            p_related_words *= flag
            p_related_words = self.softmax(p_related_words)

            _, related_word_idx = p_related_words.topk(1, 1, True, True)  # batch_size, 1
            # print(related_word_idx)
            root_node_idxs[:, i + 1] = related_word_idx.squeeze(1)
            distance_embedding_from_ck[:, i + 1, :] = distance_embedding
            new_wprob_start_from_ck[:, i + 1, :] = self.adjacent_words_embedding(related_word_idx.squeeze(1))

        return root_node_idxs[:, 1:], distance_embedding_from_ck[:, 1:]


class DecoderModel(nn.Module):
    def __init__(self, words_embeddings_from_graph, dis_embeddings_from_graph,
                 adjTable, triple_size, vocab_size, dis_size, iter_times=10, embedd_dim=512, encoder_dim=2048,
                 hidden_size=512, attention_dim=800, dropout=0.5):  # , radjTable
        super(DecoderModel, self).__init__()
        # print('Initialization of DecoderModel...')
        self.vocab_size = vocab_size
        self.triple_size = triple_size
        self.dis_size = dis_size
        self.hidden_size = hidden_size
        self.embedd_dim = embedd_dim

        self.words_embedding = nn.Embedding.from_pretrained(words_embeddings_from_graph, freeze=False)
        self.dis_embedding = nn.Embedding.from_pretrained(dis_embeddings_from_graph, freeze=False)
        self.adjacent_words_embedding = nn.Embedding.from_pretrained(adjTable)

        self.root_classifier = RootClassifier(vocab_size=vocab_size)
        self.graph_left_encoding = Graph_Encoding(vocab_size, dis_size, encoder_dim, embedd_dim, dropout, hidden_size,
                                                  iter_times, self.adjacent_words_embedding)
        self.graph_right_encoding = Graph_Encoding(vocab_size, dis_size, encoder_dim, embedd_dim, dropout, hidden_size,
                                                   iter_times, self.adjacent_words_embedding)
        self.graph_left_attention = selfAttention(embedd_dim)
        self.graph_right_attention = selfAttention(embedd_dim)
        self.generation = CaptionGeneratin(vocab_size=vocab_size, triple_size=triple_size, dis_size=dis_size,
                                           lstm_hidden_dim=hidden_size, attention_dim=attention_dim,
                                           embedd_dim=embedd_dim)
        self.init_weights()

    def init_weights(self):
        self.graph_self_attention.bias.data.fill_(0)
        self.graph_self_attention.weight.data.uniform_(-0.1, 0.1)

    def MyMultiply(self, news_idx, all_idx):
        '''
        :param news_idx: batch_size, num_related_words, *
        :param all_idx: *
        :return:
        '''
        assert news_idx.size(2) == all_idx.size(0)
        if all_idx.size(0) == self.vocab_size:
            all_embeddings = self.words_embedding(all_idx)
        else:
            all_embeddings = self.relations_embedding(all_idx)
        return torch.matmul(news_idx, all_embeddings)

    def forward(self, out_root, out_caption, encoded_captions, caption_lengths, encoded_triples, layer):

        # prepare for data
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        out_caption = out_caption[sort_ind]
        out_root = out_root[sort_ind]
        encoded_triples = encoded_triples[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        decode_lengths = (caption_lengths - 1).tolist()

        # words embeddings
        # print('words embeddings processing...')
        captions_embedding = self.words_embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        root_node_idx = self.root_classifier(out_root).squeeze(1)
        '''
        1. root
        2. root -> related_words(in dependency graph)  # batch_size, nums_related_words(id)
        3. related words -> according to global graph -> ns_words  # batch_size, num_related_words, vocab_size
        4. related words -> according to global graph -> relations  # batch_size, num_related_words, relation_size
        '''
        # print('Graph_encoding processing...')
        left_root_node_idxs, left_distance_embedding_from_ck = self.graph_left_encoding(root_node_idx, out_root,
                                                                                        self.words_embedding,
                                                                                        self.dis_embedding)
        right_root_node_idxs, right_distance_embedding_from_ck = self.graph_right_encoding(root_node_idx, out_root,
                                                                                           self.words_embedding,
                                                                                           self.dis_embedding)

        left_root_node_embeddings = self.words_embedding(left_root_node_idxs)
        right_root_node_embeddings = self.words_embedding(right_root_node_idxs)

        left_graph_embeddings = torch.add(left_root_node_embeddings,
                                          left_distance_embedding_from_ck)  # batch_size, iter_times, embedding_dim
        right_graph_embeddings = torch.add(right_root_node_embeddings,
                                           right_distance_embedding_from_ck)  # batch_size, iter_times, embedding_dim
        '''why we use self-attention'''
        left_focus_graph_embeddings = self.graph_left_attention(q=left_graph_embeddings, k=right_graph_embeddings,
                                                                v=right_graph_embeddings,
                                                                mask=None)  # batch_size, iter_times, embedding_dim
        right_focus_graph_embeddings = self.graph_right_attention(q=right_graph_embeddings, k=left_graph_embeddings,
                                                                  v=left_graph_embeddings,
                                                                  mask=None)  # batch_size, iter_times, embedding_dim
        graph_embeddings = torch.cat([left_focus_graph_embeddings, right_focus_graph_embeddings], dim=1)
        caption_predictions, ds_predictions, end_score, alphas = self.generation(out_caption, encoded_triples,
                                                                                 decode_lengths, captions_embedding,
                                                                                 graph_embeddings)

        end_score = pack_padded_sequence(end_score, [l for l in decode_lengths], batch_first=True).data
        end_score = end_score.mean() if end_score.mean() < 10 else torch.FloatTensor([1.5]).to(device)

        root_node_embedding = self.words_embedding(root_node_idx)
        hidden_vector = captions_embedding.mean(dim=1).clone().detach()  # batch_size, max_length, hidden_dim
        root_loss = (torch.LongTensor([1]).to(device) - torch.cosine_similarity(root_node_embedding, hidden_vector,
                                                                                eps=1e-8)).mean()
        return caption_predictions, ds_predictions, end_score, root_loss, encoded_captions, encoded_triples, decode_lengths, alphas, sort_ind