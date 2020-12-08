import torch.utils.data
import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
from skimage.transform import resize
import imageio
import os

# imageFolder = r'C:\Users\DELL\Desktop\flickr30k\image\\'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cap_folder = r'E:\flickr_30k\input_Captions'
ds_folder = r'E:\flickr_30k\input_DS'
data_name = 'flickr30k_5_cap_per_img_5_min_word_freq'
ModelPath = r'E:\flickr_30k\BEST_checkpoint_new61_flickr30k.pth.tar'
WordMapPath = os.path.join(cap_folder, 'WORDMAP_' + data_name + '.json')
TriplesMapPath = os.path.join(ds_folder, 'WORDMAP_' + 'DSUNIT_' + data_name + '.json')
# DPMapPath = r'C:\Users\DELL\Desktop\flickr8k\inputFile_Ds\WORDMAP_flickr8k_5_ds_per_img_5_min_word_freq.json'
TestPath = os.path.join(cap_folder, 'TEST_IMG_Filename.json')

def loadModel():
    # Load model
    checkpoint = torch.load(ModelPath, map_location='cpu')
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()
    return encoder, decoder

def processImage(image_path):
    # Read image and process
    img = imageio.imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = resize(img, (256, 256), preserve_range=True)
    img = img.astype('uint8')
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    return image

def caption_image_beam_search(encoder, decoder, image_path, word_map, ds_map):
    print(image_path)
    image = processImage(image_path)

    # Encode
    encoder_out_root, encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)
    encoder_out_root = encoder_out_root.unsqueeze(0)  # 1, encoder_dim
    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]]).to(device)  # 1, 1
    k_prev_ds = torch.LongTensor([[ds_map['<start>']]]).to(device)  # 1, 1
    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(1, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    root_node_idx = decoder.root_classifier(encoder_out_root).to(device)
    root_node_idx = root_node_idx.squeeze(1)
    # print(root_node_idx)
    left, left_dis = decoder.graph_left_encoding(root_node_idx, encoder_out_root, decoder.words_embedding, decoder.dis_embedding)
    right, right_dis = decoder.graph_right_encoding(root_node_idx, encoder_out_root, decoder.words_embedding, decoder.dis_embedding)
    left_root_node_embeddings = decoder.words_embedding(left)
    right_root_node_embeddings = decoder.words_embedding(right)
    left_words_embeddings = torch.add(left_root_node_embeddings, left_dis)  # 1, iter_times, embedding_dim
    right_words_embeddings = torch.add(right_root_node_embeddings, right_dis)  # 1, iter_times, embedding_dim

    graph_embeddings = decoder.graph_self_attention(q=left_words_embeddings, k=right_words_embeddings, v=right_words_embeddings, mask=None)
    # Start decoding
    step = 1
    h1, c1, h2, c2 = decoder.generation.init_hidden_state(encoder_out, graph_embeddings)
    # print('h1 size: {}'.format(h1.size()))
    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.words_embedding(k_prev_words[:, -1])  # (1, embed_dim)
        triple_embedding = decoder.generation.triples_embedding(k_prev_ds[:, -1])  # (1, embed_dim)
        # print('embeddings size: {}'.format(embeddings.size()))
        awe, alpha = decoder.generation.image_attention(h2, encoder_out)  # (1, num_pixels, encoder_dim), (1, hidden_size)
        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (1, enc_image_size, enc_image_size)

        gate = decoder.generation.sigmoid(decoder.generation.image_beta(h2))  # gating scalar, (1, encoder_dim)
        awe = gate * awe

        score = decoder.generation.graph_weight(torch.bmm(graph_embeddings, h1.unsqueeze(-1)).squeeze(-1))  # 1, iter_times
        rws = (graph_embeddings * score.unsqueeze(-1)).sum(dim=1)  # batch_size, embedding_dim
        # print('rws size: {}'.format(rws.size()))
        h1, c1 = decoder.generation.decode_ds_step(torch.cat([triple_embedding, rws], dim=1), (h1, c1))  # (1, decoder_dim)

        h2, c2 = decoder.generation.decode_step(torch.cat([embeddings, h1, awe], dim=1), (h2, c2))  # (1, decoder_dim)

        preds = decoder.generation.fc_vocab(h2)  # (1, vocab_size)
        ds_pred = decoder.generation.fc_trisize(h1)  # (1, vocab_size)
        scores = F.log_softmax(preds, dim=1)
        ds_scores = F.log_softmax(ds_pred, dim=1)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        # top_k_scores 是分数，top_k_words是下标

        top_k_scores, next_word_inds = scores.topk(1, 1, True, True)  # (1)
        dstop_k_scores, next_triple_inds = ds_scores.topk(1, 1, True, True)  # (1)

        # print('next word: {}, next word size: {}'.format(next_word_inds, next_word_inds.size()))
        # print('next triple: {}, next triple size: {}'.format(next_triple_inds, next_triple_inds.size()))

        # print('k prev size: {}'.format(k_prev_words.size()))
        # Add new words to sequences, alphas
        k_prev_words = torch.cat([k_prev_words, next_word_inds], dim=1)  # (1, step+1)
        seqs_alpha = torch.cat([seqs_alpha, alpha.unsqueeze(1)], dim=1)  # (1, step+1, enc_image_size, enc_image_size)
        k_prev_ds = torch.cat([k_prev_ds, next_triple_inds], dim=1)
        # print('{}th test, captions:  {}\n, triples:  {}'.format(step, k_prev_words, k_prev_ds))
        step += 1
        if next_word_inds.item() == word_map['<end>']:
            break
    k_prev_words = k_prev_words.squeeze().tolist()
    k_prev_ds = k_prev_ds.squeeze().tolist()
    seqs_alpha = seqs_alpha.squeeze().tolist()
    root_node_idx = root_node_idx.squeeze()
    print('Length of the captions: {}, the triples: {}'.format(len(k_prev_words), len(k_prev_ds)))
    return k_prev_words, seqs_alpha, k_prev_ds, root_node_idx

def getTestImageData():
    data_head = r"E:\flickr_30k\flickr30k-images\\"
    with open(TestPath, 'r') as f:
        image_absolute_paths = json.load(f)
    new_absolute_path = []
    for path in image_absolute_paths:
        new_absolute_path.append(data_head + path.split("\\")[-1])
    print('Test for Flickr30K, size: {}'.format(len(new_absolute_path)))
    return new_absolute_path

def writeData(url, hyp):
    # write hyp.txt
    with open(url, 'w') as f:
        for X in hyp:
            for l, x in enumerate(X):
                if l == len(X) - 1:
                    f.write(x + '\n')
                else:
                    f.write(x + ' ')

if __name__ == '__main__':

    # load the model
    encoder, decoder = loadModel()
    # load the testimage and the trainimage hash
    testImageFile = getTestImageData()

    # Load word map (word2ix)
    with open(WordMapPath, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
    with open(TriplesMapPath, 'r') as j:
        ds_map = json.load(j)
    rev_ds_map = {v: k for k, v in ds_map.items()}

    # reference
    hyp = []
    for i, testImagePath in enumerate(testImageFile):
        print("the {}th image".format(i))

        # Encode, decode with attention and beam search
        print("Inference ...")
        seq, alphas, ds_seq, root_node_idx = caption_image_beam_search(encoder, decoder, testImagePath, word_map, ds_map)
        print('The ROOT word is: {}'.format(rev_word_map[root_node_idx.item()]))
        triples = [rev_ds_map[ind] for ind in ds_seq[1:len(ds_seq) - 1]]
        print("Triples: ", end=" ")
        temp = []
        for j, x in enumerate(triples):
            temp.append('<' + x + '>')
            if j == len(triples) - 1:
                print('<' + x + '>', end="\n")
            else:
                print('<' + x + '>', end=", ")
        words = [rev_word_map[ind] for ind in seq[1:len(seq) - 1]]
        print("Captions: ", end=" ")
        for j, x in enumerate(words):
            if j == len(words) - 1:
                print(x, end="\n")
            else:
                print(x, end=" ")
        hyp.append(words)

    # write hyp.txt
    writeData('./flickr30k_hyp/duallstm6_hyp.txt', hyp)