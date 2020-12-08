import os
import numpy as np
import h5py
import json
from skimage.transform import resize
import imageio
from tqdm import tqdm
from collections import Counter
from random import seed, choice
import torch
from torch import nn


def create_input_files(dataset, karpathy_json_path, ds_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)
    with open(ds_json_path, 'r') as f:
        ds_data = json.load(f)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    train_image_ds_units = []
    train_image_ds_relations = []
    val_image_paths = []
    val_image_captions = []
    val_image_ds_units = []
    val_image_ds_relations = []
    test_image_paths = []
    test_image_captions = []
    test_image_ds_units = []
    test_image_ds_relations = []
    word_freq = Counter()
    relations_freq = Counter()
    ds_units_freq = Counter()

    for img, ds in zip(data['image'], ds_data["ds"]):
        captions, dsunits, dsrelations = [], [], []
        for c in img['sentence']:
            # Update word frequency
            for i in range(len(c['token'])):
                # each 5 captions； ['I', 'like', 'you', '.']
                # dslis: [[triples], [triples]]
                dslis, triples, relations = ds["dslis"][i], [], []
                for triple in dslis:
                    triples.append(triple[0]+ ',' + str(triple[1])+ ',' + str(triple[2]))
                    relations.append(triple[0])
                relations_freq.update(relations)
                ds_units_freq.update(triples)
                word_freq.update(c['token'][i])
                if len(c['token'][i]) <= max_len:
                    captions.append(c['token'][i])
                    dsunits.append(triples)
                    dsrelations.append(relations)
        # captions: [['I', 'like', 'you', '.'], ['I', 'love', 'you', '.']]
        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
            image_folder, img['filename'])

        if img['split'] in {'TRAIN', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
            train_image_ds_units.append(dsunits)
            train_image_ds_relations.append(dsrelations)
        elif img['split'] in {'DEV'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
            val_image_ds_units.append(dsunits)
            val_image_ds_relations.append(dsrelations)
        elif img['split'] in {'TEST'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)
            test_image_ds_units.append(dsunits)
            test_image_ds_relations.append(dsrelations)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions) == len(train_image_ds_units) == len(train_image_ds_relations)
    assert len(val_image_paths) == len(val_image_captions) == len(val_image_ds_units) == len(val_image_ds_relations)
    assert len(test_image_paths) == len(test_image_captions) == len(test_image_ds_units) == len(test_image_ds_relations)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create ds units map
    ds_units = [w for w in ds_units_freq.keys() if ds_units_freq[w] > min_word_freq]
    ds_units_map = {k: v + 1 for v, k in enumerate(ds_units)}
    ds_units_map['<unk>'] = len(ds_units_map) + 1
    ds_units_map['<start>'] = len(ds_units_map) + 1
    ds_units_map['<end>'] = len(ds_units_map) + 1
    ds_units_map['<pad>'] = 0

    # Create ds units map
    relations = [w for w in relations_freq.keys()]
    relations_map = {k: v + 1 for v, k in enumerate(relations)}
    relations_map['<unk>'] = len(relations_map) + 1
    relations_map['<start>'] = len(relations_map) + 1
    relations_map['<end>'] = len(relations_map) + 1
    relations_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder[0], 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Save word map to a JSON
    with open(os.path.join(output_folder[1], 'TRIPLEMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(ds_units_map, j)

    # Save word map to a JSON
    with open(os.path.join(output_folder[1], 'RELATIONMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(relations_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, imds_units, imds_relations, split in [(train_image_paths, train_image_captions, train_image_ds_units, train_image_ds_relations, 'TRAIN'),
                                   (val_image_paths, val_image_captions, val_image_ds_units, val_image_ds_relations, 'DEV'),
                                   (test_image_paths, test_image_captions, test_image_ds_units, test_image_ds_relations, 'TEST')]:

        with h5py.File(os.path.join(output_folder[0], split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions, caplens, ds_units, ds_relations = [], [], [], []

            for i, path in enumerate(tqdm(impaths)):
                captions, caps, ds_unitses, dss, ds_rel, dss_rel = [], [], [], [], [], []
                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    for _ in range(captions_per_image - len(imcaps[i])):
                        cap = choice(imcaps[i])
                        dss.append(imds_units[i][imcaps[i].index(cap)])
                        dss_rel.append(imds_relations[i][imcaps[i].index(cap)])
                        caps.append(cap)
                    captions = imcaps[i] + caps
                    ds_unitses = imds_units[i] + dss
                    ds_rel = imds_relations[i] + dss_rel
                else:
                    captions, ds_unitses, ds_rel = imcaps[i][:5], imds_units[i][:5], imds_relations[i][:5]
                #print('captions: {}'.format(captions))
                #print('ds_unitses: {}'.format(ds_unitses))
                # Sanity check
                assert len(captions) == captions_per_image == len(ds_unitses) == len(ds_rel)

                # Read images
                img = imageio.imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = resize(img, (256, 256), preserve_range=True)
                img = img.astype('uint8')
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255
                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
                    id_ds_units = [ds_units_map['<start>']] + [ds_units_map.get(dsu, ds_units_map['<unk>']) for dsu in ds_unitses[j]]\
                                  + [ds_units_map['<end>']] + [ds_units_map['<pad>']] * (max_len - len(c))
                    id_ds_rels = [relations_map['<start>']] + [relations_map.get(dsr, relations_map['<unk>']) for dsr in ds_rel[j]] \
                                  + [relations_map['<end>']] + [relations_map['<pad>']] * (max_len - len(c))
                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)
                    ds_units.append(id_ds_units)
                    ds_relations.append(id_ds_rels)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(ds_relations) == len(enc_captions) == len(caplens) == len(ds_units)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder[0], split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder[0], split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)

            with open(os.path.join(output_folder[1], split + '_DSTRIPLES_' + base_filename + '.json'), 'w') as j:
                json.dump(ds_units, j)

            with open(os.path.join(output_folder[1], split + '_DSRELES_' + base_filename + '.json'), 'w') as j:
                json.dump(ds_relations, j)

def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, model, image_extractor, optimizer, image_optimizer, bleu4, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'our_model': model,
             'image_extractor': image_extractor,
             'our_optimizer': optimizer,
             'image_optimizer': image_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)

class selfAttention(nn.Module):

    def __init__(self, decoder_dim, head_size=8, dropout_rate=0.1):
        super(selfAttention, self).__init__()

        self.head_size = head_size
        self.att_size = att_size = decoder_dim // head_size
        self.linear_q = nn.Linear(decoder_dim, head_size * att_size, bias=False)
        self.linear_k = nn.Linear(decoder_dim, head_size * att_size, bias=False)
        self.linear_v = nn.Linear(decoder_dim, head_size * att_size, bias=False)
        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        self.d = att_size ** -0.5
        self.att_dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(head_size * att_size, decoder_dim, bias=False)

    def forward(self, q, k, v, mask):
        orig_q_size = q.size()
        batch_size = q.size(0)  # 32
        d_k = self.att_size  # 512 / 8

        query = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)  # batch_size, 52, 8, 512/8
        query = query.transpose(1, 2)  # batch_size, 8, 52, 512/8
        key = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)  # batch_size, 52, 8, 512/8
        key = key.transpose(1, 2).transpose(2, 3)  # batch_size, 8, 512/8, 52
        value = self.linear_v(v).view(batch_size, -1, self.head_size, d_k)  # batch_size, 52, 8, 512/8
        value = value.transpose(1, 2)  # batch_size, 8, 52, 512/8

        query.mul_(self.d)
        x = torch.matmul(query, key)  # batch_size, 8, 50, 50
        if mask: x.masked_fill_(mask.unsqueeze(1), 1e-9)
        x = torch.softmax(x, dim=3)  # batch_size, 8, 50, 50
        x = self.att_dropout(x)
        x = x.matmul(value)  # batch_size, 8, 52, 512/8

        # contiguous()
        x = x.transpose(1, 2).contiguous()  # batch_size, 52, 8, 512/8
        x = x.view(batch_size, -1, self.head_size * d_k)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x