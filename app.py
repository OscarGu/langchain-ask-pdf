from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader

# add openAI key input
def get_api_key():
    input_text = st.text_input(label="OpenAI API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input")
    return input_text

OPENAI_API_KEY = get_api_key()

# add Pinecone key input
def get_pinecone_key():
    input_text = st.text_input(label="Pinecone API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="pinecone_api_key_input")
    return input_text

PINECONE_API_KEY = get_pinecone_key()


def main():
    load_dotenv()
    #st.set_page_config(page_title="Ask your File")
    st.header("Ask your PDF 💬")
    
   
    # upload file
    pdf = st.file_uploader("Upload your PDF/Text")
    
    # extract the text
    if pdf is not None:
      loader = TextLoader(pdf)
      documents = loader.load()
    
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      docs = text_splitter.split_documents(documents)
      
      # create embeddings
      embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

      index_name = "doc-chat"
      docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)


      
      # show user input
      user_question = st.text_input("Ask a question about your PDF:")
      if user_question:
        docs = docsearch.similarity_search(user_question)
                
        if OPENAI_API_KEY:        
            llm = OpenAI(openai_api_key=OPENAI_API_KEY)
            chain = load_qa_chain(llm, chain_type="map_reduce")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)
            
            st.write(response)
    

if __name__ == '__main__':
    main()
