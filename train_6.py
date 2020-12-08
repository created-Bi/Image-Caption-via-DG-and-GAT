import time
import json
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils import clip_grad_norm_
from model_dual_lstm_6 import DecoderModel, ImageExtractor
from datasets import *
from utils import *
from Config import *
from nltk.translate.bleu_score import corpus_bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# URLs
data_folder_captions = '/home/liumaofu//bjq/DATASET/Flickr30k/input_Captions'
data_folder_ds = '/home/liumaofu//bjq/DATASET/Flickr30k/input_DS'
data_name = 'flickr30k_5_cap_per_img_5_min_word_freq'
root_data_name = 'new6_flickr30k'
URL_DIS_Embedding = '/home/liumaofu/bjq/ICBKG/DIS_Embedding_512_flickr30k.json'
URL_WORDS_Embedding = '/home/liumaofu/bjq/ICBKG/WORDS_Embedding_512_flickr30k.json'

ADJURL_node2node = '/home/liumaofu/bjq/DATASET/Flickr30k/graph2AdjacencyTable_flickr30k.json'
ADJURL_node2rel = '/home/liumaofu/bjq/DATASET/Flickr30k/graph2RelAdjacencyTable_flickr30k.json'
# load some configurations
conf = COFIG()
start_epoch = 0
epochs_since_improvement = 0
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches


def read_embeddings(url):
    with open(url, 'r') as j:
        result = json.load(j)
    embeddings = []
    for k, v in result.items():
        embeddings.append(v)
    embeddings = torch.FloatTensor(embeddings).to(device)
    return embeddings


def main():
    global conf, epochs_since_improvement, start_epoch, best_bleu4, data_folder, data_name, word_map, rev_word_map, ds_units_map
    # print('step-0')
    # load the dependency adjacency table (vocab_size, vocab_size), and obtain (3, vocab_size)
    with open(ADJURL_node2node, 'r') as f:
        dependency_words_tab = json.load(f)
    dependency_words_tab = torch.FloatTensor(dependency_words_tab).to(device)
    with open(ADJURL_node2rel, 'r') as f:
        dependency_rel_tab = json.load(f)
    dependency_rel_tab = torch.FloatTensor(dependency_rel_tab).to(device)
    # print('step-1')
    # Read word map
    word_map_file = os.path.join(data_folder_captions, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
    relations_map_file = os.path.join(data_folder_ds, 'WORDMAP_DSUNIT_' + data_name + '.json')
    with open(relations_map_file, 'r') as j:
        triples_map = json.load(j)

    # print('step-2')
    # Read words embedding
    words_embeddings_from_DSG = read_embeddings(URL_WORDS_Embedding).to(device)
    dis_embeddings_from_DSG = read_embeddings(URL_DIS_Embedding).to(device)
    # print('step-3')
    decoder = DecoderModel(words_embeddings_from_graph=words_embeddings_from_DSG,
                           dis_embeddings_from_graph=dis_embeddings_from_DSG,
                           adjTable=dependency_words_tab, triple_size=len(triples_map), vocab_size=len(word_map),
                           dis_size=dis_embeddings_from_DSG.size(0),
                           iter_times=conf.iter_times, embedd_dim=conf.embedd_dim, encoder_dim=conf.encoder_dim,
                           hidden_size=conf.lstm_hidden_dim, attention_dim=conf.attention_dim,
                           dropout=conf.dropout)
    if conf.checkpoint is None:
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=conf.lr)
        # encoded_image_size = 14
        encoder = ImageExtractor(out_image_size=conf.out_image_size)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=conf.image_extractor_lr)
    else:
        checkpoint = torch.load(conf.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        encoder = checkpoint['encoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder_optimizer = checkpoint['encoder_optimizer']

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    #  load data (image, root words, adjacency matrix, dslis, captions)
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder_captions=data_folder_captions, data_folder_ds=data_folder_ds,
                       Capdata_name=data_name, split='TRAIN', transform=transforms.Compose([normalize])),
        batch_size=conf.batch_size, shuffle=True, num_workers=5, pin_memory=True)
    dev_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder_captions=data_folder_captions, data_folder_ds=data_folder_ds,
                       Capdata_name=data_name, split='DEV', transform=transforms.Compose([normalize])),
        batch_size=conf.batch_size, shuffle=True, num_workers=5, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, conf.epoch):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              model=decoder,
              image_extractor=encoder,
              cross_entropy=criterion,
              our_optimizer=decoder_optimizer,
              image_optimizer=encoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=dev_loader,
                                model=decoder,
                                image_extractor=encoder,
                                cross_entropy=criterion)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(root_data_name, epoch, epochs_since_improvement, decoder, encoder, decoder_optimizer,
                        encoder_optimizer, recent_bleu4, is_best)


# net train
def train(train_loader, model, image_extractor, cross_entropy, our_optimizer,
          image_optimizer, epoch):  # , global_dependency_words, global_dependency_relations
    model.train()  # train mode (dropout and batchnorm is used)
    image_extractor.train()
    # print('TRAIN')
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()
    for i, (img, caption, caplen, triples_list) in enumerate(train_loader):  # , rootWords, dsunites
        """
        :param img: batch_size, 3, 256, 256
        :param dsunits: batch_size, 50
        :param caption, caplen: batch_size, 50, batch_size, 1
        :param relations, distances:batch_size, 50
        """
        img = img.to(device)
        caption = caption.to(device)
        caplen = caplen.to(device)
        triples_list = triples_list.to(device)

        # print('model running...')
        encoder_out_root, encoder_out_caption = image_extractor(img)
        caption_predictions, ds_predictions, end_score, root_loss, encoded_captions, encoded_triples, decode_lengths, alphas, sort_ind = model(
            encoder_out_root, encoder_out_caption,
            caption, caplen, triples_list, conf.layer)
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        caps_targets = encoded_captions[:, 1:]
        encoded_triples = encoded_triples[:, 1:]
        cap_scores = pack_padded_sequence(caption_predictions, decode_lengths, batch_first=True).data
        cap_targets = pack_padded_sequence(caps_targets, decode_lengths, batch_first=True).data
        triples_scores = pack_padded_sequence(ds_predictions, decode_lengths, batch_first=True).data
        triples_targets = pack_padded_sequence(encoded_triples, decode_lengths, batch_first=True).data

        # Calculate loss
        loss_cross = cross_entropy(cap_scores, cap_targets)
        loss_cross_np = cross_entropy(triples_scores, triples_targets)
        loss_img = conf.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        loss = loss_cross + loss_cross_np + loss_img + end_score + root_loss

        # backward
        loss.backward()

        # Clip gradients
        if conf.grad_clip is not None:
            clip_grad_norm_(model.parameters(), conf.grad_clip)
            clip_grad_norm_(image_extractor.parameters(), conf.grad_clip)

        # Update weights
        our_optimizer.step()
        if image_optimizer is not None:
            image_optimizer.step()

        # optimizer
        our_optimizer.zero_grad()
        if image_optimizer is not None:
            image_optimizer.zero_grad()

        top5 = accuracy(cap_scores, cap_targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
            print('Loss - cross: {}, Loss - img: {}, Loss - end: {}'.format(loss_cross.item() + loss_cross_np.item(),
                                                                            loss_img.item(), end_score.item()))


# val
def validate(val_loader, model, image_extractor, cross_entropy):
    model.eval()
    image_extractor.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)
    image_idx = -1

    with torch.no_grad():
        for i, (img, caption, caplen, allcaps, triples_list, image_id) in enumerate(val_loader):
            if i == 0: image_idx = image_id
            img = img.to(device)
            caption = caption.to(device)
            caplen = caplen.to(device)
            triples_list = triples_list.to(device)

            encoder_out_root, encoder_out_caption = image_extractor(img)
            caption_predictions, ds_predictions, end_score, root_loss, encoded_captions, encoded_triples, decode_lengths, alphas, sort_ind = model(
                encoder_out_root, encoder_out_caption, caption, caplen, triples_list, conf.layer)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            caps_targets = encoded_captions[:, 1:]
            scores_copy = caption_predictions.clone()
            encoded_triples = encoded_triples[:, 1:]

            cap_scores = pack_padded_sequence(caption_predictions, decode_lengths, batch_first=True).data
            cap_targets = pack_padded_sequence(caps_targets, decode_lengths, batch_first=True).data
            triples_scores = pack_padded_sequence(ds_predictions, decode_lengths, batch_first=True).data
            triples_targets = pack_padded_sequence(encoded_triples, decode_lengths, batch_first=True).data

            # Calculate loss
            loss_cross = cross_entropy(cap_scores, cap_targets)
            loss_cross_np = cross_entropy(triples_scores, triples_targets)
            loss_img = conf.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            loss = loss_cross + loss_cross_np + loss_img + end_score + root_loss

            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(cap_scores, cap_targets, 5)  # 32 * vocab_size
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))
                print(
                    'Loss - cross: {}, Loss - img: {}, Loss - end: {}'.format(loss_cross.item() + loss_cross_np.item(),
                                                                              loss_img.item(), end_score.item()))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References batch_size, max_length, 5
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses batch_size, max_length, 1
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

            # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))
        referencess = []
        for referrence in references[0]:
            referencess.append([rev_word_map[ind] for ind in referrence[1:len(referrence) - 1]])

        hyp = [rev_word_map[ind] for ind in hypotheses[0][:len(hypotheses[0]) - 1]]
        print('References_size: {}, References: {}\nHypotheses_size: {}, Hypotheses: {}'.format(len(referencess),
                                                                                                referencess, len(hyp),
                                                                                                hyp))
        print('Image IDX is: {}'.format(image_idx))

    return bleu4


if __name__ == '__main__':
    main()