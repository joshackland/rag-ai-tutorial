from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


def prompt_wthout_rag(prompt):
    llm = ChatOllama(
        model="llama3.2:1b",
    )

    response = llm.invoke(
        input=[
            ("user", prompt)
        ],
    )
    
    return response.content

def prompt_with_rag(prompt):
    with open("openai_deep_research.txt", "r", encoding="utf-8") as f:
        file_text = f.read()
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.split_text(file_text)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db_chroma = Chroma.from_texts(chunks, embeddings, persist_directory="rag_db")

    docs_chroma = db_chroma.similarity_search_with_score(prompt)
    context_text = "\n\n".join([doc.page_content for doc, _ in docs_chroma])

    PROMPT_TEMPLATE = """
    Respond to the prompt based on only the following context:
    {context}

    Answer the following prompt based on the above context: 
    {prompt}
    """

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, prompt=prompt)

    llm = ChatOllama(
        model="llama3.2:1b",
    )

    response = llm.invoke(
        input=[
            ("user", prompt)
        ],
    )
    
    return response.content


if __name__ == "__main__":
    prompt1 = "What is your cutoff date?"
    prompt2 = "Explain to me in detail what OpenAI Deep Research is."

    print("----Llama 3.2:1b Cuttoff Date----")
    print(prompt_wthout_rag(prompt1))
    print("\n\n\n----Prompt Without RAG----")
    print(prompt_wthout_rag(prompt2))
    print("\n\n\n----Prompt With RAG----")
    print(prompt_with_rag(prompt2))