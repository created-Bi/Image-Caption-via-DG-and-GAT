# Configuration during the process of generation

class COFIG():
    def __init__(self, lr=4e-4, image_extractor_lr=1e-3, alpha_c=1., alpha_rd=0.7, grad_clip=2., epoch=20, start_epoch=0,
                 weight_decay=0.5, batch_size=32, dropout=0.5, out_image_size=14, encoder_dim=2048, belta=0.5,
                 embedd_dim=512, lstm_hidden_dim=512, attention_dim=800, nums_related_words=25, max_seq_length=50,
                 iter_times=50, layer=5, checkpoint=None):
        super(COFIG, self).__init__()
        self.lr = lr
        self.image_extractor_lr = image_extractor_lr
        self.dropout = dropout
        self.alpha_c = alpha_c
        self.alpha_rd = alpha_rd
        self.belta = belta
        self.grad_clip = grad_clip
        self.epoch = epoch
        self.start_epoch = start_epoch
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.out_image_size = out_image_size
        self.encoder_dim = encoder_dim
        self.nums_related_words = nums_related_words
        self.embedd_dim = embedd_dim
        self.attention_dim = attention_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.max_seq_length = max_seq_length
        self.iter_times = iter_times
        self.layer = layer
        self.checkpoint = checkpoint