import dotenv
import os
import streamlit as st

dotenv.load_dotenv('/.env')
HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
model_id = 'TheBloke/Llama-2-7B-Chat-GGUF'

# Configure for 4-bit quantization (optimizes model deployment)
bnb_config = BitsAndBytesConfig(
    bnb_4bit_compute_dtype = 'float16',
    bnb_4bit_quant_type='nf4',
    load_in_4bit=True,
)

# Load model configuration
model_config = AutoConfig.from_pretrained(
    model_id,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=model_config,
    device_map='auto',
    quantization_config=bnb_config,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
)

# Set model into evaluation mode (optimizes inference)
model.eval()

# Set up the text-generation pipeline
pipe = pipeline(
    model=model,
    task='text-generation',
    tokenizer=tokenizer
)

from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=pipe)

from langchain.prompts.prompt import PromptTemplate

# Template using jinja2 syntax
template = """
<s>[INST] <<SYS>>
The following is a friendly conversation between a human and an AI. 
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know.
Please be concise.
<</SYS>>

Current conversation:
{{ history }}

{% if history %}
    <s>[INST] Human: {{ input }} [/INST] AI: </s>
{% else %}
    Human: {{ input }} [/INST] AI: </s>
{% endif %} 
"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template,
    template_format="jinja2"
)

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Initialize the conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    prompt=prompt,
    verbose=False
)

# Start the conversation
def predict(message: str, history: str):
    response = conversation.predict(input=message)

    return response

answer = predict("Hello", "")
if answer:
  st.write(answer)
