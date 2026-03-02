# Conext RAG

Sistema de perguntas e respostas sobre documentos PDF utilizando Retrieval-Augmented Generation (RAG) com avaliação de qualidade através do framework RAGAS.

## Funcionalidades

- Carregamento e indexação de documentos PDF
- Busca semântica via embeddings do HuggingFace
- Geração de respostas contextualizadas usando LLM
- Avaliação automática de qualidade com métricas RAGAS (faithfulness, answer relevancy, context precision)

## Requisitos

- Python 3.8+
- Conta com acesso a API de LLM compatível com OpenAI
- Token do HuggingFace

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

Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:

```
DO_BASE_URL=sua_url_base_api
DO_API_KEY=sua_chave_api
DO_MODEL=nome_do_modelo
HF_TOKEN=seu_token_huggingface
```

## Uso

```bash
python main.py
```

O sistema irá:
1. Indexar o PDF especificado
2. Solicitar uma pergunta
3. Retornar a resposta baseada no contexto do documento
4. Exibir métricas de avaliação da resposta

## Tecnologias

- LangChain: orchestração do pipeline RAG
- ChromaDB: armazenamento vetorial
- HuggingFace: embeddings (sentence-transformers/all-MiniLM-L6-v2)
- RAGAS: avaliação de qualidade do RAG
- PyPDF: processamento de PDFs
