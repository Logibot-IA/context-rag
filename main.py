import os
from dotenv import load_dotenv

from itertools import count

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGCHAIN_API_KEY"]     = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]     = os.getenv("LANGCHAIN_PROJECT", "benchmark-context-rag")

BASE_URL = os.getenv("DO_BASE_URL")
API_KEY = os.getenv("DO_API_KEY")
MODEL = os.getenv("DO_MODEL")
HF_TOKEN = os.getenv("HF_TOKEN")

test_queries = [
    # FÁCEIS
    "O que significa ‘lógica de programação’ em palavras simples?",
    "De um jeito bem direto: o que é um algoritmo?",
    "Qual é a diferença entre constante e variável?",
    "Pra que serve o comando ‘leia’ em um algoritmo?",

    # MÉDIAS
    "O que é um comando de atribuição e por que o tipo do dado precisa ser compatível com o tipo da variável?",
    "O que são operadores aritméticos (como +, -, * e /) e pra que eles servem?",
    "Pra que servem os operadores relacionais numa expressão?",

    # DIFÍCEIS
    "O que é uma ‘expressão lógica’?",
    "Em uma repetição, o que é um contador e como ele é incrementado?",
    "Como funciona a repetição ‘repita ... até’ e o que ela garante sobre a execução do bloco?"
]

ground_truths = [
    # FÁCEIS
    "Lógica de programação é o uso correto das leis do pensamento, da ‘ordem da razão’ e de processos formais de raciocínio e simbolização na programação de computadores, com o objetivo de produzir soluções logicamente válidas e coerentes para resolver problemas.",
    "Um algoritmo é uma sequência de passos bem definidos que têm por objetivo solucionar um determinado problema.",
    "Um dado é constante quando não sofre variação durante a execução do algoritmo: seu valor permanece constante do início ao fim (e também em execuções diferentes ao longo do tempo). Já um dado é variável quando pode ser alterado em algum instante durante a execução do algoritmo, ou quando seu valor depende da execução em um certo momento ou circunstância.",
    "O comando de entrada de dados ‘leia’ é usado para que o algoritmo receba os dados de que precisa: ele tem a finalidade de atribuir o dado fornecido à variável identificada, seguindo a sintaxe leia(identificador) (por exemplo, leia(X) ou leia(A, XPTO, NOTA)).",

    # MÉDIAS
    "Um comando de atribuição permite fornecer um valor a uma variável. O tipo do dado atribuído deve ser compatível com o tipo da variável: por exemplo, só se pode atribuir um valor lógico a uma variável declarada como do tipo lógico.",
    "Operadores aritméticos são o conjunto de símbolos que representam as operações básicas da matemática (por exemplo: + para adição, - para subtração, * para multiplicação e / para divisão). Para potenciação e radiciação, o livro indica o uso das palavras‑chave pot e rad.",
    "Operadores relacionais são usados para realizar comparações entre dois valores de mesmo tipo primitivo. Esses valores podem ser constantes, variáveis ou expressões aritméticas, e esses operadores são comuns na construção de equações.",

    # DIFÍCEIS
    "Uma expressão lógica é aquela cujos operadores são lógicos ou relacionais e cujos operandos são relações, variáveis ou constantes do tipo lógico.",
    "Um contador é um modo de contagem feito com a ajuda de uma variável com um valor inicial, que é incrementada a cada repetição. Incrementar significa somar um valor constante (normalmente 1) a cada repetição.",
    "A estrutura de repetição ‘repita ... até’ permite que um bloco (ou ação primitiva) seja repetido até que uma determinada condição seja verdadeira. Pela sintaxe da estrutura, o bloco é executado pelo menos uma vez, independentemente da validade inicial da condição."
]

PERSIST_DIR = "./chroma_context_db"

def build_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        collection_name="context_collection",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    if vectordb._collection.count() == 0:
        loader = DirectoryLoader("./docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        print(f"Adicionando {len(chunks)} chunks ao Chroma em batches...")
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vectordb.add_documents(documents=batch)
            print(f"  {min(i + batch_size, len(chunks))}/{len(chunks)} chunks adicionados")
        print("Ingestão concluída!")
    else:
        print(f"Coleção existente com {vectordb._collection.count()} chunks. Pulando ingestão.")

    return vectordb, embeddings

vectordb, embeddings = build_vectorstore()
retriever = vectordb.as_retriever()
print(f"Vectorstore pronto: {vectordb._collection.count()} chunks indexados.")

llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL,
    temperature=0
)

def context_rag(query, retriever, llm, top_k=5):
    docs = retriever.invoke(query)
    selected_docs = docs[:top_k]
    contexts = [d.page_content for d in selected_docs]
    context_text = "".join(contexts)

    prompt = f"""
    Você deve responder usando SOMENTE o contexto fornecido.

    Contexto:
    {context_text}

    Pergunta:
    {query}

    Se a resposta não estiver no contexto, diga:
    "A informação não está presente no contexto."
    """

    response = llm.invoke(prompt).content
    return response, contexts

from langsmith import traceable

@traceable(name="context-rag-query", run_type="chain")
def context_rag_traced(query, retriever, llm, top_k=5):
    return context_rag(query, retriever, llm, top_k)

print("Coletando respostas para avaliação RAGAS...")
ragas_data = []

for i, query in enumerate(test_queries):
    print(f"  [{i+1}/{len(test_queries)}] {query}")
    answer, contexts = context_rag_traced(query, retriever, llm)
    ragas_data.append({
        "question": query,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truths[i]
    })

def run_ragas(ragas_data, llm, embeddings):
    dataset = Dataset.from_list(ragas_data)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings
    )

    print("=== RESULTADOS RAGAS ===")
    print(result)

    df = result.to_pandas()
    print("Detalhes por query:")
    print(df.to_string())

    return result

def salvar(df, nome_base="context-rag"):
    if not hasattr(salvar, "_results_dir"):
        base_dir = "results"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=False)
            salvar._results_dir = base_dir
        else:
            for n in count(2):
                candidate = f"{base_dir}_{n}"
                if not os.path.exists(candidate):
                    os.makedirs(candidate, exist_ok=False)
                    salvar._results_dir = candidate
                    break

        print(f"Resultados desta execução serão salvos em: {salvar._results_dir}")

    os.makedirs(salvar._results_dir, exist_ok=True)
    for i in count(1):
        nome = os.path.join(salvar._results_dir, f"{nome_base}_{i}.csv")
        if not os.path.exists(nome):
            df.to_csv(nome, index=False, encoding="utf-8-sig", sep=";")
            print(f"Salvo em: {nome}")
            break

for i in range(15):
    print(f"=== RODADA {i+1} ===")
    result = run_ragas(ragas_data, llm, embeddings)
    salvar(result.to_pandas(), nome_base=f"context-rag-run-{i+1}")