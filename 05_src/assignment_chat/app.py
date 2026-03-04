#### Preamble ####
# Purpose: Gradio chat interface for the assignment chatbot.
#          Run with: cd 05_src && python -m assignment_chat.app
# Author: Mike Cowan
# Date: 27 February 2026
# Contact: m.cowan@utoronto.ca
# License: MIT

import gradio as gr
from dotenv import load_dotenv

from assignment_chat.agent import cowboy_chat
from utils.logger import get_logger

_logs = get_logger(__name__)
load_dotenv('.secrets')

#### Entry Point ####

chat = gr.ChatInterface(
    fn=cowboy_chat,
    type="messages",
)

if __name__ == "__main__":
    _logs.info('Starting assignment chat...')
    chat.launch()
