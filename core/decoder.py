from bert4torch.generation import AutoRegressiveDecoder

class ArticleSummaryDecoder(AutoRegressiveDecoder):
    """
    Decoder for article summarization using AutoRegressiveDecoder.
    """
    def __init__(self, model, tokenizer, **kwargs):
        super(ArticleSummaryDecoder, self).__init__(**kwargs)
        self.model = model
        self.tokenizer = tokenizer

    @AutoRegressiveDecoder.wraps(default_rtype='logits')
    def predict(self, inputs, output_ids, states=None):
        """
        Predict the next token logits.
        
        Args:
            inputs: List containing input token ids.
            output_ids: Current generated output token ids.
            states: Optional states.
            
        Returns:
            Logits for the next token.
        """
        token_ids = inputs[0] 
        logits = self.model.predict([token_ids, output_ids])
        return logits[-1][:, -1, :]

    def generate(self, text, maxlen=512, topk=4):
        """
        Generate summary for the given text.
        
        Args:
            text: Input text to summarize.
            maxlen: Maximum length of the input text.
            topk: Top-k for beam search.
            
        Returns:
            Decoded summary string.
        """
        token_ids, _ = self.tokenizer.encode(text, maxlen=maxlen)
        output_ids = self.beam_search([token_ids], top_k=topk)
        output_ids = output_ids[0]
        return self.tokenizer.decode(output_ids)