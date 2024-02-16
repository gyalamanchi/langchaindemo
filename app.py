import chainlit as cl
from customllm import CustomLLM
from customembedder import CustomEmbedder

from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema.document import Document

template = "You are an assistant that helps users."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
#human_template = """Context: {context}
#Question: {question}"""
human_template = """Question: {question}  Context: {context}"""
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

question = "Who won the SuperBowl in 2018?"
context = "The Philadelphia Eagles won the SuperBowl in 2018"
template = """Question: {question}
Context: {context}
Answer: Let's think step by step."""
llm = CustomLLM()
#context = db.similarity_search("Tell me about Putin")
#context = db.as_retriever().get_relevant_documents(context)
#print(context)

@cl.on_chat_start
async def main():    
    elements = [
    cl.Image(name="image1", display="inline", path="./IMG_0624.jpg")
    ]
    await cl.Message(content="Hey there this is Goldie Happy Times AI Demo, dont mind the pumpkin!", elements=elements).send()
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!",
            accept=["text/plain"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Decode the file
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    text = file.content.decode("utf-8")
    print(text)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    sentence_transformer_ef = CustomEmbedder()
    db = Chroma.from_documents(docs, sentence_transformer_ef)

    cl.user_session.set("db", db)
    
    llm_chain = LLMChain(llm=llm,
                          prompt=chat_prompt,
                          verbose=True)
    
    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain
    db = cl.user_session.get("db")  # type: LLMChain
    context = db.as_retriever().get_relevant_documents(message.content)
    print(context)
    answer = llm_chain({"question": message.content, "context": context})
    #answer = StrOutputParser.parse(answer)
    #print(answer)
    await cl.Message(answer["text"]).send()