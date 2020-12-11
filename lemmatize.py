import nltk

lemmatizer = nltk.stem.WordNetLemmatizer()

with open("unked.txt", "r") as file1:
    poems = file1.read().split("\n")

poems = [poem.split() for poem in poems]


poems = [nltk.pos_tag(poem) for poem in poems]

print(poems)

with open("data/unklemma.txt", "w") as file2:
    for poem in poems:
        for word, pos in poem:
            pos = pos[0].lower()
            pos = pos if pos in ['a', 'r', 'n', 'v'] else None
            if not pos:
                lemma = word
            else:
                lemma = lemmatizer.lemmatize(word, pos)
            file2.write(lemma + " ")
        file2.write("\n")

