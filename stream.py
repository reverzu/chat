from langchain.llms import CTransformers
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML", model_file = 'llama-2-7b-chat.ggmlv3.q8_0.bin', callbacks=[StreamingStdOutCallbackHandler()])

template = """
[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Your answers are always brief.
<</SYS>>
{text}[/INST]
"""

prompt = PromptTemplate(template=template, input_variables=["text"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

for response in llm("Why some days are terrible?", stream=True):
  print(response)
  st.markdown(response)
