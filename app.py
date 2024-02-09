import chainlit as cl
from customllm import CustomLLM
from customembedder import CustomEmbedder

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser


loader = TextLoader("./sotu2.txt")
documents = loader.load()

# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
#sentence_transformer_ef = SentenceTransformerEmbeddings(model_name="/opt/app/workspace/ai_projects/revature/all-MiniLM-L6-v2")
sentence_transformer_ef = CustomEmbedder()

# load it into Chroma
#client = chromadb.HttpClient(host="localhost", port=8000)
#chroma_client = chromadb.PersistentClient(path="/Users/goldieyalamanchi/workspace/chroma")
#db = Chroma(client=client, collection_name="my_collection", embedding_function=sentence_transformer_ef)
#db = Chroma.from_documents(docs, sentence_transformer_ef)
db = Chroma.from_documents(docs, sentence_transformer_ef)

question = "Who won the SuperBowl in 2018?"
context = "The Philadelphia Eagles won the SuperBowl in 2018"
template = """Question: {question}
Context: {context}
Answer: Let's think step by step."""
llm = CustomLLM()
#context = db.similarity_search("Tell me about Putin")
#context = db.as_retriever().get_relevant_documents(context)
print(context)

@cl.on_chat_start
def main():    
    prompt = PromptTemplate(template=template, input_variables=["context","question"])
    llm_chain = LLMChain(llm=llm,
                          prompt=prompt,
                          verbose=True)
    
    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain
    context = db.similarity_search(message.content)
    #context = db.as_retriever().get_relevant_documents(message.content)
    #print(context)
    answer = llm_chain({"question": message.content, "context": context})
    #answer = StrOutputParser.parse(answer)
    #print(answer)
    await cl.Message(answer).send()