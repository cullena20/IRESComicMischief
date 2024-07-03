from torch import nn

class FeatureEncoding(nn.Module):
    def __init__(self):
        super(FeatureEncoding, self).__init__()
        self.rnn_units = 512
        self.embedding_dim = 768
        dropout = 0.2

        self.rnn_audio = nn.LSTM(128, self.rnn_units, num_layers=2, bidirectional=True, batch_first=True)
        self.rnn_audio_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.rnn_units*2, eps=1e-5),
        ) 
        self.rnn_img = nn.LSTM(1024, self.rnn_units, num_layers=2, bidirectional=True, batch_first=True)
        self.rnn_img_drop_norm = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(self.rnn_units*2, eps=1e-5),
        ) 
        
        self.sequential_audio = nn.Sequential(
            nn.Linear(self.rnn_units*2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(0.3),
        )
        self.sequential_image = nn.Sequential(
            nn.Linear(self.rnn_units*2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(0.3),
        )

    def forward(self, sentences, image, audio):
        self.encoded_text, _ = self.bert(sentences)[-2:]
        
        rnn_img_encoded, (hid, ct) = self.rnn_img(image)
        rnn_img_encoded = self.rnn_img_drop_norm(rnn_img_encoded)
        rnn_audio_encoded, (hid_audio, ct_audio) = self.rnn_audio(audio)
        rnn_audio_encoded = self.rnn_audio_drop_norm(rnn_audio_encoded)

        self.encoded_img = self.sequential_image(rnn_img_encoded), 
        self.encoded_audio = self.sequential_audio(rnn_audio_encoded)