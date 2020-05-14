from tensorflow.keras.preprocessing.text import Tokenizer
# Tokenizer identifies the works as tockens which helps in easy identify the word
sentences = [
    'i am vishal',
    'I, love my cat',
    'she is beautiful!'
]
#object
tokenizer = Tokenizer(num_words = 100) #max .it also takes frequency into con.
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
