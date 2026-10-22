# Agente de Consulta de Clima

Chat de terminal, em português, que consulta o **clima atual de qualquer cidade do mundo** usando um agente LangChain + modelo local no Ollama. Sem chave de API.

```
Clima no terminal - bonsai-2 (thinking desligado)
Pergunte o clima de qualquer lugar. Ex.: "como está o tempo em Lisboa?"
Comandos: "sair" para encerrar, "limpar" para apagar o histórico.

Você: qual o clima em Recife?
  [consultando via get_weather...]

Bonsai: O clima em Recife é principalmente limpo, com temperatura de 27,0 °C
(sensação de 30,0 °C), umidade de 75% e vento de 14,2 km/h.

Você: E qual é a umidade lá?
Bonsai: A umidade em Recife é de 75%.
```

## Como funciona

O agente recebe a pergunta em linguagem natural, decide chamar a ferramenta `get_weather` e só então redige a resposta — sem inventar dados meteorológicos:

```
Você digita
    │
    ▼
create_agent (LangGraph)
    │
    ├─► modelo (Ollama) ──► tool_call: get_weather(city="Recife")
    │                              │
    │                              ▼
    │                       Open-Meteo API
    │                  (geocodificação + clima atual)
    │                              │
    │                              ▼
    │                       ToolMessage (texto do clima)
    │
    └─► resposta final em streaming ──► terminal
```

| Componente | Papel |
|---|---|
| `ChatOllama(model="bonsai-2", reasoning=False)` | LLM local; `reasoning=False` envia `think: false` ao Ollama |
| `get_weather(city, country="")` | Ferramenta: resolve o nome do lugar e busca o clima atual |
| `InMemorySaver` + `thread_id` | Memória da conversa (permite "e a umidade lá?") |
| `stream_mode="messages"` | Resposta impressa token a token |

## Requisitos

- **Python 3.10+** (testado com 3.12)
- **Ollama** instalado e rodando (`ollama serve`)
- Um modelo com suporte a *tool calling* baixado no Ollama. O padrão é o `bonsai-2`; para usar outro, ajuste a constante `MODEL` em `main.py` (ex.: `qwen3`, `llama3.1`).
- Acesso à internet (as APIs do Open-Meteo são públicas e não exigem cadastro)

## Instalação

```bash
git clone https://github.com/Dione-b/agente-consulta-clima.git
cd agente-consulta-clima

python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

Garanta que o modelo está disponível:

```bash
ollama list          # confira se o modelo aparece
ollama serve         # se ainda não estiver rodando
```

## Uso

```bash
python main.py
```

Exemplos de perguntas:

| Pergunta | O que acontece |
|---|---|
| `como está o tempo em Recife?` | geocodifica e busca o clima atual |
| `vai chover em Lisboa?` | consulta precipitação atual |
| `qual a temperatura em Springfield, Estados Unidos?` | usa a dica de país para desambiguar |
| `e a umidade lá?` | reaproveita o contexto da conversa |

Comandos do chat:

| Comando | Efeito |
|---|---|
| `sair`, `exit`, `quit`, `q` | encerra o programa |
| `limpar`, `reset`, `novo` | apaga o histórico e começa nova sessão |
| `Ctrl+C` | interrompe a resposta atual, sem fechar |
| `Ctrl+D` | encerra (EOF) |

## Dados retornados

A ferramenta devolve um bloco de texto que o modelo usa como base da resposta:

```
Local: Lisboa, Distrito de Lisboa, Portugal (38.73, -9.15)
Horário local: 2026-10-22 23:00 (Europe/Lisbon)
Condição: nublado
Temperatura: 14.3 °C (sensação de 13.8 °C)
Umidade: 75%
Vento: 3.6 km/h
Precipitação: 0.0 mm
```

- **Geocodificação:** [`geocoding-api.open-meteo.com`](https://open-meteo.com/en/docs/geocoding-api) — resolve cidade → latitude/longitude, em português.
- **Clima atual:** [`api.open-meteo.com/v1/forecast`](https://open-meteo.com/en/docs) — com `timezone=auto` para o horário local.
- **Códigos WMO** são traduzidos por um mapa interno (`WMO_CODES`) para descrições como "garoa congelante forte" ou "tempestade com granizo".

## Configuração

Constantes no topo de `main.py`:

| Constante | Padrão | Descrição |
|---|---|---|
| `MODEL` | `"bonsai-2"` | modelo do Ollama |
| `TIMEOUT` | `15` | timeout (s) das chamadas HTTP |
| `SYSTEM_PROMPT` | — | instrui o uso obrigatório da ferramenta e respostas curtas em pt-BR |

O modelo é instanciado com `reasoning=False` (thinking desligado) e `temperature=0.2`.

## Estrutura

```
.
├── main.py           # agente, ferramenta de clima e loop de terminal
├── requirements.txt  # dependências Python
├── .gitignore
└── README.md
```

## Limitações

- **Somente clima atual.** A API suporta previsão (`daily=...`), mas ainda não está integrada — perguntas como "vai chover amanhã?" não têm dados reais.
- **Histórico em memória e sem limite.** `InMemorySaver` guarda tudo em RAM e o contexto cresce a cada mensagem; em sessões longas vale usar `SummarizationMiddleware`.
- **Unidades fixas** em °C e km/h.
- **Modelos pequenos** podem ocasionalmente ignorar o resultado da ferramenta; o `SYSTEM_PROMPT` e a `temperature` baixa reduzem, mas não eliminam isso.

## Licença

Sem licença definida.
