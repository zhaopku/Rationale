class Sample:
    def __init__(self, data, words, steps, label, length, id):
        self.input_ = data[0:steps]
        self.sentence = words[0:steps]
        self.length = length
        self.label = label
        self.id = id


class Batch:
    def __init__(self, samples):
        self.samples = samples
        self.batch_size = len(samples)
