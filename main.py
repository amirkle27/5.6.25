from fastapi import FastAPI
from markdown_it.common.entities import entities
from pydantic import BaseModel
from spacy.lang.el.tokenizer_exceptions import token
from spacy.matcher import PhraseMatcher
from starlette.responses import HTMLResponse
import bl
import spacy

nlp = spacy.load("en_core_web_sm")

# Create FastAPI instance
app = FastAPI(title="Example API", description="A simple FastAPI with Swagger UI", version="1.0")

# Define a Pydantic model for input data
class Item(BaseModel):
    name: str
    price: float
    in_stock: bool = True

# Define a simple GET endpoint
@app.get("/welcome")
def read_root():
    return {"message": "Welcome to the FastAPI example!"}

# Define a POST endpoint
@app.get("/get-text-and-label/text/")
def get_text_and_label(text: str):
    doc = nlp(text)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    return entities


###

@app.get("/get-person-in-text/text")
def get_person_ent(text:str):
    doc = nlp(text)
    entities = [{"Person":ent.text} for ent in doc.ents if ent.label_ == "PERSON"]
    return entities

@app.get("/get-lemma-in-text/text")
def get_lemma(text:str):
    doc = nlp(text)
    lemmas = [{f"{token.text:7} ->": token.lemma_} for token in doc ]
    return lemmas



# Print all named entities along with their labels


###
@app.get("/not-stop-words/text")
def get_not_stop_words(text:str):
    doc = nlp(text)
    not_stop_words = [{token.text} for token in doc if not token.is_stop]
    return f"Not Stop Words in Text: {not_stop_words}"


@app.get("/get-stop-words/text")
def get_stop_words(text:str):
    nlp.vocab["powerful"].is_stop = True
    doc = nlp(text)

    stop_words = [{token.text} for token in doc if token.is_stop]
    return f"stop_words are: {stop_words}"
