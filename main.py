# pip install -qU langchain langchain-ollama
"""Chat de clima no terminal, usando Ollama + LangChain.

Pergunte o clima de qualquer cidade em linguagem natural, por exemplo:
    - "como está o tempo em Recife?"
    - "vai chover em Lisboa hoje?"
    - "e em Tóquio?"

O agente chama a ferramenta `get_weather`, que consulta a API pública
Open-Meteo (geocodificação + clima atual, sem necessidade de chave).

Comandos: sair | exit | quit | q  -> encerra
           limpar | reset | novo  -> apaga o histórico da conversa
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

MODEL = "bonsai-2"
TIMEOUT = 15  # segundos
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

SYSTEM_PROMPT = (
    "Você é um assistente de clima. Sempre use a ferramenta get_weather para "
    "responder perguntas sobre o tempo e nunca invente dados meteorológicos. "
    "Responda em português do Brasil, de forma curta e direta."
)

# Códigos WMO (https://open-meteo.com/en/docs) -> descrição em pt-BR
WMO_CODES = {
    0: "céu limpo",
    1: "principalmente limpo",
    2: "parcialmente nublado",
    3: "nublado",
    45: "neblina",
    48: "neblina com geada",
    51: "garoa fraca",
    53: "garoa moderada",
    55: "garoa forte",
    56: "garoa congelante fraca",
    57: "garoa congelante forte",
    61: "chuva fraca",
    63: "chuva moderada",
    65: "chuva forte",
    66: "chuva congelante fraca",
    67: "chuva congelante forte",
    71: "neve fraca",
    73: "neve moderada",
    75: "neve forte",
    77: "grãos de neve",
    80: "pancadas de chuva fracas",
    81: "pancadas de chuva moderadas",
    82: "pancadas de chuva violentas",
    85: "pancadas de neve fracas",
    86: "pancadas de neve fortes",
    95: "tempestade",
    96: "tempestade com granizo fraco",
    99: "tempestade com granizo forte",
}


def _fetch_json(url: str, params: dict[str, object]) -> dict:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": "langchain-weather-demo/1.0"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.load(response)


@lru_cache(maxsize=256)
def _geocode(city: str, country: str = "") -> dict | None:
    """Resolve o nome de um lugar para latitude/longitude."""
    city, _, hint = city.partition(",")
    hint = (country or hint).strip()

    data = _fetch_json(
        GEO_URL,
        {"name": city.strip(), "count": 5, "language": "pt", "format": "json"},
    )
    results = data.get("results") or []
    if not results:
        return None

    if hint:
        low = hint.casefold()
        results.sort(
            key=lambda r: 0
            if low in (r.get("country") or "").casefold()
            or low == (r.get("country_code") or "").casefold()
            else 1
        )
    return results[0]


def _num(value: object, unit: str = "", digits: int = 1) -> str:
    if value is None:
        return "n/d"
    return f"{float(value):.{digits}f}{unit}"


def get_weather(city: str, country: str = "") -> str:
    """Clima atual de qualquer cidade do mundo.

    Args:
        city: Nome da cidade, ex.: "Recife", "São Paulo", "Lisboa", "Tóquio".
        country: Opcional, para desambiguar: nome ou sigla do país, ex.: "Brasil", "BR".
    """
    try:
        place = _geocode(city, country)
        if place is None:
            return (
                f"Não encontrei o local '{city}'. "
                "Tente novamente com o nome da cidade (e o país, se necessário)."
            )

        current = _fetch_json(
            FORECAST_URL,
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": ",".join(
                    [
                        "temperature_2m",
                        "relative_humidity_2m",
                        "apparent_temperature",
                        "precipitation",
                        "weather_code",
                        "wind_speed_10m",
                    ]
                ),
                "timezone": "auto",
            },
        ).get("current") or {}

    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
        return f"Não foi possível consultar o serviço de clima agora ({exc})."

    if not current:
        return f"O serviço de clima não retornou dados para '{city}'."

    location = ", ".join(
        part for part in (place.get("name"), place.get("admin1"), place.get("country")) if part
    )
    condition = WMO_CODES.get(current.get("weather_code"), "condição desconhecida")
    local_time = (current.get("time") or "").replace("T", " ")

    return "\n".join(
        [
            f"Local: {location} ({_num(place.get('latitude'), '', 2)}, "
            f"{_num(place.get('longitude'), '', 2)})",
            f"Horário local: {local_time} ({place.get('timezone') or 'fuso local'})",
            f"Condição: {condition}",
            f"Temperatura: {_num(current.get('temperature_2m'), ' °C')} "
            f"(sensação de {_num(current.get('apparent_temperature'), ' °C')})",
            f"Umidade: {_num(current.get('relative_humidity_2m'), '%', 0)}",
            f"Vento: {_num(current.get('wind_speed_10m'), ' km/h')}",
            f"Precipitação: {_num(current.get('precipitation'), ' mm')}",
        ]
    )


def _responder(agent, pergunta: str, config: dict) -> None:
    """Envia a pergunta e imprime a resposta em streaming."""
    anunciadas: set[str] = set()
    escreveu = False

    for trecho, meta in agent.stream(
        {"messages": [{"role": "user", "content": pergunta}]},
        config,
        stream_mode="messages",
    ):
        if meta.get("langgraph_node") != "model" or not isinstance(trecho, AIMessageChunk):
            continue

        for chamada in trecho.tool_call_chunks:
            nome = chamada.get("name")
            if nome and nome not in anunciadas:
                anunciadas.add(nome)
                print(f"  [consultando via {nome}...]", flush=True)

        if not trecho.text:
            continue
        if not escreveu:
            print("\nBonsai: ", end="", flush=True)
            escreveu = True
        print(trecho.text, end="", flush=True)

    if escreveu:
        print()
        return

    # Nenhum texto foi transmitido: mostra o que ficou no estado do agente.
    mensagens = agent.get_state(config).values.get("messages", [])
    ultimo = mensagens[-1].text.strip() if mensagens else ""
    print(f"\nBonsai: {ultimo or '(o modelo não retornou texto)'}")


def main() -> None:
    agent = create_agent(
        model=ChatOllama(model=MODEL, reasoning=False, temperature=0.2),
        tools=[get_weather],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )

    session = 0
    print(f"Clima no terminal - {MODEL} (thinking desligado)")
    print('Pergunte o clima de qualquer lugar. Ex.: "como está o tempo em Lisboa?"')
    print('Comandos: "sair" para encerrar, "limpar" para apagar o histórico.')

    while True:
        try:
            pergunta = input("\nVocê: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAté a próxima!")
            break

        if not pergunta:
            continue

        comando = pergunta.casefold()
        if comando in {"sair", "exit", "quit", "q"}:
            print("Até a próxima!")
            break
        if comando in {"limpar", "reset", "novo"}:
            session += 1
            print("Histórico da conversa apagado.")
            continue

        config = {"configurable": {"thread_id": f"clima-{session}"}}
        try:
            _responder(agent, pergunta, config)
        except KeyboardInterrupt:
            print("\n[interrompido]")
        except Exception as exc:  # mantém o chat vivo mesmo com falhas
            print(f"\n[erro] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
