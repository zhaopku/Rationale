class Sample:
	def __init__(self, id=None, words=None, word_ids=None, length=None, label=None):
		# id is file name
		self.id = id

		# padded
		self.words = words
		self.word_ids = word_ids

		# True length
		self.length = length

		# republican of democrats
		self.label = label

