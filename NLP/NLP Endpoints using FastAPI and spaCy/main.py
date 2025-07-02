from fastapi import FastAPI
from markdown_it.common.entities import entities
from pydantic import BaseModel
from spacy.lang.el.tokenizer_exceptions import token
from spacy.matcher import PhraseMatcher
from starlette.responses import HTMLResponse
import bl
import spacy

nlp = spacy.load("en_core_web_sm")

app = FastAPI(title="Example API", description="A simple FastAPI with Swagger UI", version="1.0")

@app.get("/welcome")
def read_root():
    return {"message": "Welcome to the FastAPI example!"}

###########################  1  ###########################

@app.get("/get-text-and-label/text/")
def get_text_and_label(text: str):
    entities = bl.get_text_and_label(nlp(text))
    return entities

###########################  2  ###########################

@app.get("/get-person-in-text/text")
def get_person_ent(text:str):
    person_entities = bl.get_person_ents(nlp(text))
    return person_entities

###########################  3  ###########################

@app.get("/get-lemma-in-text/text")
def get_lemma(text:str):
    lemmas = bl.get_lemmas(nlp(text))
    return lemmas

###########################  4  ###########################

@app.get("/not-stop-words/text")
def get_not_stop_words(text:str):
    not_stop_words = bl.not_stop(nlp(text))
    return f"Not Stop Words in Text: {not_stop_words}"

###########################  5  ###########################

@app.get("/get-stop-words/text")
def get_stop_words(text:str):
    stop_words = bl.get_stop_words(nlp(text))
    return f"stop_words are: {stop_words}"

###########################  6  ###########################
from spacy.matcher import PhraseMatcher

@app.get("/phrasematcher/text")
def get_phrases(text: str):
    phrase_matches = bl.phrase_match(nlp(text))
    return {"matches": phrase_matches}

###########################  7  ###########################

@app.get("/get-token-details/text")
def get_token_details(text:str):
    token_details = bl.get_t_det(nlp(text))
    return token_details

###########################  8  ###########################

@app.get("/custom-sentence-split/text")
def split_sentences(text: str):
    sentences = bl.split_sen(nlp(text))
    return {"sentences": sentences}

###########################  #  ###########################

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=9000, reload=True)
