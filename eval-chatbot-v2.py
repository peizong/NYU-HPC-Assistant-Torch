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


# ---------------------------
# 1. Load environment
# ---------------------------
load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in your environment or .env file")


# ---------------------------
# 2. Download raw text file
# ---------------------------
url = "https://raw.githubusercontent.com/hwchase17/chroma-langchain/master/state_of_the_union.txt"
res = requests.get(url, timeout=30)
res.raise_for_status()

with open("state_of_the_union.txt", "w", encoding="utf-8") as f:
    f.write(res.text)


# ---------------------------
# 3. Load and split documents
# ---------------------------
loader = TextLoader("./state_of_the_union.txt", encoding="utf-8")
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)


# ---------------------------
# 4. Connect to embedded Weaviate
# ---------------------------
client = weaviate.connect_to_embedded(
    version="1.27.0",
    headers={
        "X-OpenAI-Api-Key": OPENAI_API_KEY
    }
)

try:
    # ---------------------------
    # 5. Build vector store
    # ---------------------------
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = WeaviateVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,   # note: singular 'embedding'
        client=client,
        index_name="StateOfUnionDemo"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # ---------------------------
    # 6. Define LLM and prompt
    # ---------------------------
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    template = """You are an assistant for question-answering tasks.
Use the following retrieved context to answer the question.
If you do not know the answer, say that you do not know.
Use at most two sentences and keep the answer concise.

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

    # ---------------------------
    # 7. Evaluation set
    # ---------------------------
    questions = [
        "What did the president say about Justice Breyer?",
        "What did the president say about Intel's CEO?",
        "What did the president say about gun violence?",
    ]

    ground_truth = [
        "The president said that Justice Breyer has dedicated his life to serving the country and thanked him for his service.",
        "The president said that Pat Gelsinger said Intel is ready to increase its investment to $100 billion.",
        "The president asked Congress to pass proven measures to reduce gun violence.",
    ]

    answers = []
    contexts = []

    for query in questions:
        answer = rag_chain.invoke(query)
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
    df.to_csv("rag_evaluation_results.csv", index=False)
    print(df)

finally:
    client.close()
