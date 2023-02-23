import spacy
from spacy import displacy

# write a function to perform named entity recognition
nlp = spacy.load('en_core_web_sm')
def ner(text):
    doc = nlp(text)
    # print(doc.ents)
    return [(ent.text, ent.label_) for ent in doc.ents]

# write a function to get augmented references from the named entity recognition

def get_augmented_reference_text(reference_text):
  #vineet-irfan
    doc = nlp(reference_text)
    #displacy.render(doc, style="ent",jupyter=True)
    for word in doc.ents:
      #print(word.text)
      new_word = word.text.replace('.', '')
      reference_text = reference_text.replace(word.text, new_word)
    return reference_text