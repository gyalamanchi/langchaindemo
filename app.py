import chainlit as cl
from customllm import CustomLLM

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

question = "Who won the FIFA World Cup in the year 1994? "

template = """Question: {question}

Answer: Let's think step by step."""

llm = CustomLLM()

@cl.on_chat_start
def main():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    conv_chain = LLMChain(llm=llm,
                          prompt=prompt,
                          verbose=True)
    
    cl.user_session.set("llm_chain", conv_chain)

@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    answer = llm_chain({"question": message.content})
    print(answer)
    await cl.Message(answer).send()