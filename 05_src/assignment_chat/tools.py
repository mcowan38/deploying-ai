#### Preamble ####
# Purpose: Tool functions and JSON schemas for the assignment chatbot.
#          Three services: trivia API, Father Brown semantic search,
#          and a numexpr calculator.
# Author: Mike Cowan
# Date: 27 February 2026
# Contact: m.cowan@utoronto.ca
# License: MIT
# Pre-requisites:
# - chromadb PersistentClient data at ./assignment_chat/chroma_data
#   (run setup_db.py first)
# References:
# - [https://opentdb.com/api_config.php]
# - [https://docs.trychroma.com/docs/run-chroma/persistent-client]

#### Workspace Setup ####

import os
import json
import html
import random
import math

import requests
import numexpr
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from utils.logger import get_logger

_logs = get_logger(__name__)
load_dotenv(".env")
load_dotenv(".secrets")

#### ChromaDB Connection ####
embedding_fn = OpenAIEmbeddingFunction(
    api_key="any value",
    model_name="text-embedding-3-small",
    api_base="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
    default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY")},
)

chroma = chromadb.PersistentClient(path="./assignment_chat/chroma_data")
collection = chroma.get_collection(
    name="father_brown", embedding_function=embedding_fn
)


#### Tool Functions ####

def get_trivia(difficulty="medium"):
    _logs.debug(f"get_trivia: difficulty={difficulty}")
    params = {"amount": 1, "type": "multiple", "difficulty": difficulty.lower()}
    resp = requests.get("https://opentdb.com/api.php", params=params)
    results = resp.json().get("results", [])
    if not results:
        return "No trivia found. Try again."

    q = results[0]
    question = html.unescape(q["question"])
    correct = html.unescape(q["correct_answer"])
    choices = [html.unescape(a) for a in q["incorrect_answers"]] + [correct]
    random.shuffle(choices)
    labels = "ABCD"

    return (
        f"Category: {html.unescape(q['category'])}\n"
        f"Difficulty: {q['difficulty'].capitalize()}\n\n"
        f"Question: {question}\n\n"
        + "\n".join(f"  {labels[i]}. {c}" for i, c in enumerate(choices))
        + f"\n\nCorrect Answer: {correct}"
    )


def search_text(query, n_results=3):
    _logs.debug(f"search_text: query={query!r}")
    results = collection.query(query_texts=[query], n_results=n_results)
    if not results["documents"][0]:
        return "No passages found."

    passages = []
    for idx, doc in enumerate(results["documents"][0]):
        chunk_id = results["ids"][0][idx]
        passages.append(f"[{chunk_id}] {doc}")
    return "\n---\n".join(passages)


def calculate(expression):
    _logs.debug(f"calculate: expression={expression!r}")
    try:
        local_dict = {"pi": math.pi, "e": math.e}
        result = numexpr.evaluate(
            expression.strip(),
            global_dict={},
            local_dict=local_dict,
        )
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


#### OpenAI Tool Schemas ####

TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "get_trivia",
        "description": "Fetches a random multiple-choice trivia question.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "difficulty": {
                    "type": "string",
                    "description": "easy, medium, or hard",
                    "enum": ["easy", "medium", "hard"],
                },
            },
            "required": ["difficulty"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "search_text",
        "description": "Searches Father Brown stories for relevant passages.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in the text.",
                },
                "n_results": {
                    "type": "number",
                    "description": "Number of passages to return (1-5).",
                },
            },
            "required": ["query", "n_results"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "calculate",
        "description": "Evaluates a math expression. Examples: '2 + 3 * 4', 'sqrt(144)', 'pi * 3**2'.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression to evaluate.",
                },
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
]
