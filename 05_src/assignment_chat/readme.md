# The Cowboy

Chatbot with a cowboy personality that does trivia, searches Father Brown stories, and solves math. Extends the cowboy persona from assignment 1. Uses the OpenAI Responses API with tool dispatch instead of LangGraph.

## Services

+ Trivia (`get_trivia` in `tools.py`) calls the Open Trivia Database API and formats the response as a quiz.
+ Literary search (`search_text` in `tools.py`) uses ChromaDB PersistentClient over chesterton.txt. Run `setup_db.py` once to build the embeddings.
+ Calculator (`calculate` in `tools.py`) evaluates math expressions with numexpr directly.

## How to Run

From the repo root:

    source deploying-ai-env/bin/activate
    cd 05_src
    python -m assignment_chat.setup_db   # one-time: builds ChromaDB embeddings
    python -m assignment_chat.app        # starts Gradio at http://127.0.0.1:7860

## Decisions

+ Kept using the OpenAI SDK from assignment 1 instead of switching to LangGraph
+ Put all three tools in one file with JSON schemas instead of separate files with LangChain decorators
+ Used numexpr directly for the calculator instead of routing through math_tools.py
+ Used PersistentClient instead of Docker ChromaDB to keep it simpler
