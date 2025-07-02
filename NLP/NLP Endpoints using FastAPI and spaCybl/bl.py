from fastapi import FastAPI
from pydantic import BaseModel
import spacy
from spacy.matcher import PhraseMatcher
from main import nlp

###########################  1  ###########################

def get_text_and_label(doc):
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

###########################  2  ###########################

def get_person_ents(doc):
    doc = [{"Person":ent.text} for ent in doc.ents if ent.label_ == "PERSON"]
    return doc

###########################  3  ###########################

def get_lemmas(doc):
    doc = [{f"{token.text:7} ->": token.lemma_} for token in doc ]
    return doc

###########################  4  ###########################

def not_stop(doc):
    doc = [{token.text} for token in doc if not token.is_stop]
    return doc

###########################  5  ###########################

def get_stop_words(doc):
    nlp.vocab["powerful"].is_stop = True
    doc = [{token.text} for token in doc if token.is_stop]
    return doc

###########################  6  ###########################

def phrase_match(doc):
    matcher = PhraseMatcher(nlp.vocab)
    phrases = ["artificial intelligence", "Artificial Intelligence"]
    patterns = [nlp(p) for p in phrases]

    matcher.add("AI_PHRASE", patterns)
    matches = matcher(doc)
    results = []
    for match_id, start, end in matches:
        results.append(doc[start:end].text)
    return results

###########################  7  ###########################

def get_t_det(doc):
    details = [{"token": token.text, "token.pos": token.pos_, "description": spacy.explain(token.pos_)} for token in doc]
    return details

###########################  8  ###########################

from spacy.language import Language

@Language.component("custom_separator")
def custom_separator(doc):
    for token in doc[:-1]:
        if token.text == '^':
            doc[token.i + 1].is_sent_start = True
    return doc

nlp.add_pipe("custom_separator", before="parser")

def split_sen(doc):
    sentences = [sent.text for sent in doc.sents]
    return sentences
