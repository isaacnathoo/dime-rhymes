import tensorlayer

with open("data/kaggle2.csv", "r") as file:
    poems = file.read().split("\n")

words = []
for poem in poems:
    for word in poem.split(): words.append(word)

print(words)

data, count, dictionary, reverse_dictionary = tensorlayer.nlp.build_words_dataset(words=words, vocabulary_size=10000, printable=True, unk_key='UNK')

with open("unked.txt", "w") as file2:
    for poem in poems:
        for word in poem.split():
            if word in dictionary:
                file2.write(word + " ")
            else: file2.write("UNK ")
        file2.write("\n")
