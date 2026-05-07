# Context RAG

Sistema de perguntas e respostas sobre documentos PDF utilizando Retrieval-Augmented Generation (RAG) com avaliação de qualidade através do framework RAGAS.

## Funcionalidades

- Carregamento e indexação de documentos PDF
- Busca semântica via OpenAI Embeddings (`text-embedding-3-large`)
- Geração de respostas contextualizadas via OpenAI (`gpt-5.5` por padrão)
- Avaliação automática de qualidade com métricas RAGAS (faithfulness, answer relevancy, context precision)

## Requisitos

- Python 3.10+
- Conta na OpenAI com acesso à API
- Chave de API da OpenAI

## Ambiente Virtual

Criar e ativar ambiente virtual:

```bash
python -m venv .venv
```

Ativar no Windows:
```bash
.venv\Scripts\activate
```

Ativar no Git Bash:
```bash
source .venv/Scripts/activate
```

Desativar:
```bash
deactivate
```

## Instalação

Com o ambiente virtual ativado, instale as dependências:

```bash
pip install -r requirements.txt
```

## Configuração

Crie um arquivo `.env` na raiz do projeto a partir do `.env.example` e informe sua chave:

```
OPENAI_API_KEY=sk-sua_chave_openai
OPENAI_MODEL=gpt-5.5
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_REASONING_EFFORT=medium
CHROMA_PERSIST_DIR=./chroma_context_db_openai
CHROMA_COLLECTION_NAME=context_collection_openai
```

Ao trocar o modelo de embeddings, use um diretório novo para o ChromaDB ou remova a base vetorial antiga, porque dimensões de embeddings diferentes não podem compartilhar a mesma coleção.

## Uso

```bash
python main.py
```

O sistema irá:
1. Indexar os PDFs da pasta `docs/`
2. Executar as perguntas de benchmark configuradas em `main.py`
3. Gerar respostas usando somente os trechos recuperados
4. Exibir e salvar métricas de avaliação RAGAS

## Tecnologias

- LangChain: orchestração do pipeline RAG
- ChromaDB: armazenamento vetorial
- OpenAI: embeddings (`text-embedding-3-large`) e modelo de respostas (`gpt-5.5`)
- RAGAS: avaliação de qualidade do RAG
- PyPDF: processamento de PDFs
