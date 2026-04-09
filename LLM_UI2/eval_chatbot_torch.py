import streamlit as st
import os
from core.faisembedder import FaissEmbedder
from openai import OpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders, Portkey

#modules for evalutation

import os
import requests
from dotenv import load_dotenv, find_dotenv
from datasets import Dataset

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import weaviate
from langchain_weaviate import WeaviateVectorStore

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
###end modulues for evaluation

# Configuration Variables
MODEL_NAME = "@gpt-4o-mini/gpt-4o-mini"  # OpenAI model to use
PAGE_TITLE = "NYU HPC Assistant"
PAGE_ICON = "🤖"
WELCOME_MESSAGE = "Ask any questions about NYU's High Performance Computing resources!"
CHAT_PLACEHOLDER = "What would you like to know about NYU's HPC?"
RESULTS_COUNT = 4  # Number of similar documents to retrieve
MAX_CHAT_HISTORY = 6  # Number of recent messages to include in context

RESOURCES_FOLDER = "resources-torch"
RAG_DATA_FILE = "rag_prepared_data_nyu_hpc.csv"
FAISS_INDEX_FILE = "faiss_index.pkl"

def initialize_embedder():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resources_dir = os.path.join(script_dir, RESOURCES_FOLDER)
    rag_output = os.path.join(resources_dir, RAG_DATA_FILE)
    faiss_index_file = os.path.join(resources_dir, FAISS_INDEX_FILE)
    
    return FaissEmbedder(rag_output, index_file=faiss_index_file)

def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
    
    st.title(PAGE_TITLE)
    st.markdown(WELCOME_MESSAGE)

    # Add clear chat button
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "embedder" not in st.session_state:
        st.session_state.embedder = initialize_embedder()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input(CHAT_PLACEHOLDER):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            results = st.session_state.embedder.search(prompt, k=RESULTS_COUNT)
            context = "\n".join([result['metadata']['chunk'] for result in results])
            
            chat_history = ""
            if len(st.session_state.messages) > 0:
                recent_messages = st.session_state.messages[-MAX_CHAT_HISTORY:]
                chat_history = "\nRecent conversation:\n" + "\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in recent_messages
                ])
            
            messages = [
                {"role": "system", "content": """You are a helpful assistant specializing in NYU's High Performance Computing. 
First evaluate if the provided context contains relevant information for the question:
- If the context is relevant, prioritize this NYU-specific information in your response.
- If the context is irrelevant or only tangentially related, rely on your general knowledge instead.
- Do not mention "context", the user does not know how the code works internally.

Supplement your responses with general knowledge about HPC concepts, best practices, and technical explanations where appropriate.
Always ensure your responses are accurate and aligned with NYU's HPC environment."""},
                {"role": "user", "content": f"Context: {context}\n{chat_history}\n\nQuestion: {prompt}"}
            ]
            
            # Stream the response
            stream = st.session_state.embedder.openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                #stream=True, #this is part of the original code
            )

            # The next four lines are from Danyal's original version
            #  for chunk in stream:
            #     if chunk.choices[0].delta.content is not None:
            #         full_response += chunk.choices[0].delta.content
            #         message_placeholder.markdown(full_response + "▌")
            #if len(list(stream))==1: 
            stream=[stream]
            for chunk in stream:
                if chunk.choices[0].message.content is not None:
                    full_response += chunk.choices[0].message.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    #main()

  CHUNK_SIZE = 1000  # Size of text chunks for RAG preparation
    
  # client = Portkey(
  #         base_url = "https://ai-gateway.apps.cloud.rt.nyu.edu/v1",
  #         api_key = "H5CASawqNSqQFHctCy7NgNrOCp2n",
  # )
  client = OpenAI()
  #from langchain.schema import BaseRetriever, Document
  from langchain_core.retrievers import BaseRetriever
  from langchain_core.documents import Document
  from typing import List

  # 1. Define the Wrapper Class
  class FaissRetrieverWrapper(BaseRetriever):
    embedder: any  # This will hold your FaissEmbedder instance

    def _get_relevant_documents(self, query: str) -> List[Document]:
      # Call your custom search
      results = self.embedder.search(query, k=4) 
      #context = "\n".join([result['metadata']['chunk'] for result in results])
      print(f"\n--- DEBUG: SEARCH RESULTS FOR QUERY: '{query}' ---")
      print(f"Number of results found: {len(results)}")
      if len(results) > 0:
          print(f"First result type: {type(results[0])}")
          print(f"First result content: {results[0]}")
      print("-----------------------------------------------\n")
      return [
        Document(
            # CHANGE 'text' TO THE ACTUAL KEY FROM YOUR DATA FILE
            #page_content=res.get('chunk', ''),   #res.get('content', ''), 
            page_content=res['metadata']['chunk'], #res.get('metadata', {}).get('chunk', 'No content found'), 
            metadata=res.get('meta', {})
        ) for res in results
      ]
      #return context

  try:
    # ---------------------------
    # 5. Build vector store
    # ---------------------------
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small") #"text-embedding-3-large"
      
    # from langchain_community.embeddings import JinaEmbeddings

    # # Replace the OpenAI line with this
    # embeddings = JinaEmbeddings(
    #   jina_api_key=os.environ.get("JINA_API_KEY"), #"your_jina_api_key",
    #   model_name="jina-embeddings-v3"
    # )
      
    # vectorstore = WeaviateVectorStore.from_documents(
    #     documents=chunks,
    #     embedding=embeddings,   # note: singular 'embedding'
    #     client=client,
    #     index_name="StateOfUnionDemo"
    # )

    # retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    #retriever = initialize_embedder().search(k=RESULTS_COUNT) #search st.session_state.embedder.search(prompt, k=RESULTS_COUNT)
      
    # Create the new retriever
    custom_embedder = initialize_embedder()
    retriever = FaissRetrieverWrapper(embedder=custom_embedder)

    # ---------------------------
    # 6. Define LLM and prompt
    # ---------------------------
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    #llm = client

#     template = """You are an assistant for question-answering tasks.
# Use the following retrieved context to answer the question.
# If you do not know the answer, say that you do not know.
# Use at most two sentences and keep the answer concise.

# Question: {question}
# Context: {context}
# Answer:
# """
    template = """You are a helpful assistant specializing in NYU's High Performance Computing. 
First evaluate if the provided context contains relevant information for the question:
- If the context is relevant, prioritize this NYU-specific information in your response.
- If the context is irrelevant or only tangentially related, rely on your general knowledge instead.
- Do not mention "context", the user does not know how the code works internally.

Supplement your responses with general knowledge about HPC concepts, best practices, and technical explanations where appropriate. Always ensure your responses are accurate and aligned with NYU's HPC environment.

Question: {question}
Context: {context}
Answer:
    """
      
    prompt = ChatPromptTemplate.from_template(template)

    # Format retrieved docs into a single context string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    #debug this function and make it work
    def get_full_response(retriever):
        full_response = ""
        embedder = initialize_embedder()
        results = embedder.search(prompt, k=RESULTS_COUNT)
        context = "\n".join([result['metadata']['chunk'] for result in results])
            
        chat_history = ""
        if len(st.session_state.messages) > 0:
            recent_messages = st.session_state.messages[-MAX_CHAT_HISTORY:]
            chat_history = "\nRecent conversation:\n" + "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in recent_messages
            ])
        messages = [
                {"role": "system", "content": """You are a helpful assistant specializing in NYU's High Performance Computing. 
First evaluate if the provided context contains relevant information for the question:
- If the context is relevant, prioritize this NYU-specific information in your response.
- If the context is irrelevant or only tangentially related, rely on your general knowledge instead.
- Do not mention "context", the user does not know how the code works internally.

Supplement your responses with general knowledge about HPC concepts, best practices, and technical explanations where appropriate.
Always ensure your responses are accurate and aligned with NYU's HPC environment."""},
                {"role": "user", "content": f"Context: {context}\n{chat_history}\n\nQuestion: {prompt}"}
            ]
        # Stream the response
        stream = st.session_state.embedder.openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
                #stream=True, #this is part of the original code
        )

            # The next four lines are from Danyal's original version
            #  for chunk in stream:
            #     if chunk.choices[0].delta.content is not None:
            #         full_response += chunk.choices[0].delta.content
            #         message_placeholder.markdown(full_response + "▌")
            #if len(list(stream))==1: 
        stream=[stream]
        for chunk in stream:
            if chunk.choices[0].message.content is not None:
                full_response += chunk.choices[0].message.content
        return full_response
        
    # ---------------------------
    # 7. Evaluation set
    # ---------------------------
    questions = [
        "How to open an HPC account?",
        "How to renew an HPC account?",
        "I have a valid HPC account, but I've been unable to log in recently. What can I do?",
    ]

    ground_truth = [
        "To request an NYU HPC account, please log in to the  NYU Identity Management service and follow the link to 'Request HPC account'. We have a walkthrough of how to request an account through IIQ. If you are a student, alumni, or an external collaborator, you need an NYU faculty sponsor.",
        "Each year, non-faculty users must renew their HPC account by completing the account renewal form available through the NYU Identity Management service. See  Renewing your HPC account with IIQ for a walkthrough of the process.",
        "Check that you are using the VPN or the HPC Gateway Server (see Accessing HPC and SSH Tunneling pages for more info); Make sure you did not go over quota (both storage size and inodes number) in your home directory",
    ]

    answers = []
    contexts = []

    for query in questions:
        answer = rag_chain.invoke(query)
        print("question:", query)
        print("answer:", answer)
        retrieved_docs = retriever.invoke(query)   # preferred over get_relevant_documents
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]

        answers.append(answer)
        contexts.append(retrieved_contexts)

    # ---------------------------
    # 8. Build Ragas dataset
    # ---------------------------
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truth,   # singular column name
    }

    dataset = Dataset.from_dict(data)

    # ---------------------------
    # 9. Run evaluation
    # ---------------------------
    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=llm,
        embeddings=embeddings,
    )

    df = result.to_pandas()
    df.to_csv("torch_rag_evaluation_results.csv", index=False)
    print(df)

  finally:
    client.close()
