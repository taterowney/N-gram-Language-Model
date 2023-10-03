import numpy as np
import tiktoken

with open('data_kant.txt', 'r') as f:
    data = f.read().replace('\n', ' ').lower()

for p in """!"#$%&'()*+,-/:;<=>?@[\]^_`{|}~”""":
    data = data.replace(p, '')

def remove_whitespace(text):
    text.strip()
    return text

encoder = tiktoken.get_encoding("cl100k_base")

class NgramModelOld:
    def __init__(self, n):
        self.n = n
        self.ngram_counts = {}
        self.tokens = []
        self.tokenizer = lambda x: x.split(' ')
        self.decoder = lambda x: " ".join(x)
        self.data = data
        self.heat = 0.1
#        self.tokenizer = encoder.encode
#        self.decoder = encoder.decode

    def train(self):
        data = self.data
        self.tokens = tuple(set(data.replace(".", " <END> <START> ").split(" ")))
#        self.attention_map = np.zeros((len(self.tokens), len(self.tokens)))
        # first axis will be context, second axis will be next word
#        print(self.tokens)
        for sentence in data.split('.'):
            sentence = sentence.strip()
#            print(type(sentence.lower()))
            if sentence:
                sentence_tokens = ["<START>"]*(self.n-1) + self.tokenizer(sentence.lower()) + ["<END>"]*(self.n-1)
                ngrams = [tuple(sentence_tokens[i:i+self.n]) for i in range(len(sentence_tokens)-self.n+1)]
                for ngram in ngrams:
                    context = ngram[:-1]
                    if not context in self.ngram_counts:
                        self.ngram_counts[context] = []
                    self.ngram_counts[context].append(ngram[-1])
#                    for token in context:
#                        self.attention_map[self.tokens.index(token), self.tokens.index(ngram[-1])] += 1

    def prob_func(self, candidates):
        return np.random.choice(candidates)

    def predict(self, context):
        if not context in self.ngram_counts:
            return "Could not find context in ngram counts"
        ret = self.ngram_counts[context]
#        ret = {}
#        for c in candidates:
#            if c not in ret:
#                ret[c] = 1
#            for token in context:
#                ret[c] += self.get_frequency(token, c)**self.heat
#        val = np.random.choice(list(ret.keys()), 1, p=[ret[k]/sum(ret.values()) for k in ret.keys()])
        return np.random.choice(ret)

#    def get_frequency(self, context_token, next_token):
#        return self.attention_map[self.tokens.index(context_token), self.tokens.index(next_token)]

    def generate(self, context, max_length=100):
        generated = []
        for i in range(max_length):
            next_word = self.predict(context)
            if next_word == "<END>":
                break
            generated.append(next_word)
            context = context[1:] + (next_word,)
        return self.decoder(generated)

    def dream(self, max_length=100):
        return self.generate(tuple(["<START>"]*(self.n-1)), max_length)

class NgramModel:
    def __init__(self, n):
        self.n = n
        self.data = ["<START>"] + data.replace(".", " <END> <START> ").split(" ")
        self.tokens = list(set(self.data))
        self.tokens.remove("")
        self.tokens = tuple(self.tokens)
        self.ngram_counts = dict(zip(self.tokens, [[] for i in range(len(self.tokens))]))
        self.tokenizer = lambda x: x.split(' ')
        self.decoder = lambda x: " ".join(x)
        self.heat = 1
#        self.tokenizer = encoder.encode
#        self.decoder = encoder.decode

    def train(self):
        data = self.data
        self.attention_map = np.zeros((len(self.tokens), len(self.tokens)))
        # first axis will be context, second axis will be next word
        for i in range(self.n, len(data)):
            token = data[i]
            if not token:
                continue
            context = tuple(data[i-self.n:i])
            self.ngram_counts[token].append(context)

    def prob_func(self, candidates):
        return np.random.choice(candidates)

    # def predict(self, context):
    #     if not context in self.ngram_counts:
    #         return "Could not find context in ngram counts"
    #     candidates = self.ngram_counts[context]
    #     ret = {}
    #     for c in candidates:
    #         if c not in ret:
    #             ret[c] = 1
    #         for token in context:
    #             ret[c] += self.get_frequency(token, c)**self.heat
    #     val = np.random.choice(list(ret.keys()), 1, p=[ret[k]/sum(ret.values()) for k in ret.keys()])
    #     return val[0]

    def predict(self, context):
        candidates = {}
        for candidate in self.tokens:
            score = self.get_similarity(context, self.ngram_counts[candidate])
            candidates[candidate] = score
        return np.random.choice(list(candidates.keys()), 1, p=[candidates[k]/sum(candidates.values()) for k in candidates.keys()])[0]

    def get_similarity(self, phrase, contexts):
        total_score = 0
        for context in contexts:
            if context == phrase:
                total_score += 2.0
            for chunk_size in range(len(phrase)-1, -1, -1):
                for i in range(len(phrase)-chunk_size):
                    for j in range(len(context)-chunk_size):
                        if phrase[i:i+chunk_size] == context[j:j+chunk_size]:
                            if i == j:
                                if i+chunk_size == len(phrase):
                                    total_score += 0.0002/(chunk_size+1)
                                else:
                                    total_score += 0.0001/(chunk_size+1)
#                               else:
#                                   total_score += 0.1/(chunk_size+1)
#                                   pass
        return total_score**self.heat

    def get_frequency(self, context_token, next_token):
        return self.attention_map[self.tokens.index(context_token), self.tokens.index(next_token)]

    def generate(self, context, max_length=100):
        generated = []
        for i in range(max_length):
            next_word = self.predict(context)
            if next_word == "<END>":
                break
            generated.append(next_word)
            context = context[1:] + (next_word,)
        return self.decoder(generated)

    def dream(self, max_length=10):
        return self.generate(tuple(["<START>"]*self.n), max_length)

if __name__ == '__main__':
    model = NgramModelOld(4)
    model.train()
    print(model.dream())
#    print(model.generate(("if", "we", "wish", "to")))
