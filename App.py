from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import pandas as pd
import os
from ferramenta import *


st.set_page_config(page_title="Assistente de análise de dados com IA", layout="centered")
st.title("🦜 Assistente de análise de dados com IA")


# Descrição da ferramenta
st.info("""
Este assistente utiliza um agente, criado com Langchain, para te ajudar a explorar, analisar e visualizar dados de forma interativa. Basta fazer o upload de um arquivo CSV e você poderá:

- **Gerar relatórios automáticos**:
    - **Relatório de informações gerais**: apresenta a dimensão do DataFrame, nomes e tipos das colunas, contagem de dados nulos e duplicados, além de sugestões de tratamentos e análises adicionais.
    - **Relatório de estatísticas descritivas**: exibe valores como média, mediana, desvio padrão, mínimo e máximo; identifica possíveis outliers e sugere próximos passos com base nos padrões detectados.
- **✨Fazer perguntas simples sobre os dados**: como "Qual é a média da coluna X?", "Quantos registros existem para cada categoria da coluna Y?".
- **🤖**Criar gráficos automaticamente** com base em perguntas em linguagem natural.

Ideal para analistas, cientistas de dados e equipes que buscam agilidade e insights rápidos com apoio de IA.
""")

# Upload do CSV
st.markdown("### 📁 Faça upload do seu arquivo CSV")
arquivo_carregado = st.file_uploader("Selecione um arquivo CSV", type="csv", label_visibility="collapsed")

if arquivo_carregado:
    df = pd.read_csv(arquivo_carregado)
    st.success("Arquivo carregado com sucesso!")
    st.markdown("### Primeiras linhas do DataFrame")
    st.dataframe(df.head())

    # Obtenha a chave da API do Google
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

    # Configurando o modelo LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        api_key=GOOGLE_API_KEY,
        max_output_tokens=1024,
        temperature=0.1
    )

    # Ferramentas
    tools = criar_ferramentas(df)

    # Prompt react
    df_head = df.head().to_markdown()

    prompt_react_pt = PromptTemplate(
        input_variables=["input", "agent_scratchpad", "tools", "tool_names", "df_head"],
        template="""
    Você é um assistente que sempre responde em português.

    Você tem acesso a um dataframe pandas chamado `df`.
    Aqui estão as primeiras linhas do Dataframe, obtidas com `df.head().to_markdown()`:
    {df_head}

    Responda às seguintes perguntas da melhor forma possível.

    Para isso, você tem acesso às seguintes ferramentas:
    {tools}

    Use o seguinte formato:

    Question: a pergunta de entrada que você deve responder
    Thought: você deve sempre pensar no que fazer
    Action: a ação a ser tomada, deve ser uma das [{tool_names}]
    Action Input: a entrada para a ação
    Observation: o resultado da ação
    ... (este Thought/Action/Action Input/Observation pode se repetir N vezes)
    Thought: Agora eu sei a resposta final
    Final Answer: a resposta final para a pergunta de entrada original.
    Quando usar a ferramenta_python: formate sua resposta final de forma clara, em lista, com valores

    Comece!

    Question: {input}
    Thought: {agent_scratchpad}"""
    )
    # Agente
    agente = create_react_agent(llm=llm, tools=tools, prompt=prompt_react_pt)
    orquestrador = AgentExecutor(agent=agente,
                                tools=tools,
                                verbose=True,
                                handle_parsing_errors=True)


