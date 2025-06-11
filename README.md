# 🦜 Assistente de Análise de Dados com IA

Este projeto demonstra como agentes de IA podem ser aplicados à análise de dados, oferecendo uma interface interativa para explorar, analisar e visualizar dados de forma rápida e inteligente. Utilizando Langchain, Google Gemini e Streamlit, o assistente permite gerar relatórios automáticos, responder perguntas sobre os dados e criar gráficos a partir de arquivos CSV.

## ✨ Funcionalidades

- **Upload de arquivos CSV** para análise.
- **Relatório de informações gerais**: dimensão do DataFrame, tipos de colunas, dados nulos, duplicados e sugestões de tratamento.
- **Relatório de estatísticas descritivas**: média, mediana, desvio padrão, valores extremos, identificação de outliers e recomendações.
- **Perguntas em linguagem natural**: obtenha respostas sobre os dados, como médias, contagens e valores únicos.
- **Geração automática de gráficos**: crie visualizações a partir de comandos em português.
- **Download dos relatórios** em formato Markdown.

## 🚀 Como usar

1. **Clone o repositório:**
   ```powershell
   git clone https://github.com/seu-usuario/AgentesIA-com-Analise-de-dados.git
   cd AgentesIA-com-Analise-de-dados
   ```

2. **Instale as dependências:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure a chave da API do Google Gemini:**
   - Crie um arquivo `.env` na raiz do projeto e adicione:
     ```
     GOOGLE_API_KEY=SuaChaveAqui
     ```

4. **Execute o aplicativo:**
   ```powershell
   streamlit run App.py
   ```

5. **Acesse o app** pelo navegador no endereço exibido pelo Streamlit (geralmente http://localhost:8501).

## 🛠️ Requisitos

- Python 3.8+
- Conta e chave de API do Google Gemini
- Dependências listadas em `requirements.txt`

## 📁 Estrutura do Projeto

```
App.py                # Interface principal Streamlit
ferramenta.py         # Ferramentas de análise, relatórios e gráficos
requirements.txt      # Dependências do projeto
README.md             # Este arquivo
LICENSE               # Licença do projeto
database/             # Exemplos de dados
notebooks/            # Notebooks de experimentação
```

## 🤖 Tecnologias Utilizadas

- [Langchain](https://python.langchain.com/)
- [Google Gemini (Generative AI)](https://ai.google.dev/)
- [Streamlit](https://streamlit.io/)
- [Pandas, Matplotlib, Seaborn](https://pandas.pydata.org/)

## 👤 Autor - Ricardo Filgueiras

- [LinkeIN](https://www.linkedin.com/in/ricardo-filgueiras-b4607b232/)
- [Youtube](https://www.youtube.com/@ricardofilgueiras_datascience) 

## 📄 Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
