from bert4torch.generation import AutoRegressiveDecoder

class ArticleSummaryDecoder(AutoRegressiveDecoder):
    def __init__(self, model, tokenizer, **kwargs):
        super(ArticleSummaryDecoder, self).__init__(**kwargs)
        self.model = model
        self.tokenizer = tokenizer

    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states=None):
        token_ids = inputs[0] 
        logits = self.model.predict([token_ids, output_ids])
        return logits[-1][:, -1, :]

    def generate(self, text, maxlen=512, topk=4):
        token_ids, _ = self.tokenizer.encode(text, maxlen=maxlen)
        output_ids = self.beam_search([token_ids], top_k=topk)
        output_ids = output_ids[0]
        return self.tokenizer.decode(output_ids)