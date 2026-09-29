import math
import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch

st.set_page_config(page_title="LangGraph Multi AI Agent", page_icon="🤖", layout="wide")

load_dotenv()

# =====================================================
# STYLE + ANIMATIONS
# =====================================================

st.markdown("""
<style>

/* ---------- animated background ---------- */
.stApp {
    background: linear-gradient(-45deg, #0f172a, #1e293b, #1e1b4b, #0f172a);
    background-size: 400% 400%;
    animation: bgMove 18s ease infinite;
}
@keyframes bgMove {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ---------- floating bubbles ---------- */
.bubbles { position: fixed; inset: 0; overflow: hidden; pointer-events: none; z-index: 0; }
.bubbles span {
    position: absolute; bottom: -80px; border-radius: 50%;
    background: radial-gradient(circle, rgba(99,102,241,.35), rgba(56,189,248,.08));
    animation: rise linear infinite;
}
.bubbles span:nth-child(1) { left: 8%;  width: 40px; height: 40px; animation-duration: 16s; }
.bubbles span:nth-child(2) { left: 22%; width: 70px; height: 70px; animation-duration: 22s; animation-delay: 3s; }
.bubbles span:nth-child(3) { left: 45%; width: 30px; height: 30px; animation-duration: 14s; animation-delay: 6s; }
.bubbles span:nth-child(4) { left: 63%; width: 90px; height: 90px; animation-duration: 26s; animation-delay: 1s; }
.bubbles span:nth-child(5) { left: 80%; width: 50px; height: 50px; animation-duration: 18s; animation-delay: 8s; }
.bubbles span:nth-child(6) { left: 92%; width: 34px; height: 34px; animation-duration: 15s; animation-delay: 4s; }
@keyframes rise {
    0%   { transform: translateY(0) scale(1);   opacity: 0; }
    10%  { opacity: 1; }
    100% { transform: translateY(-115vh) scale(1.4); opacity: 0; }
}

.block-container { position: relative; z-index: 1; }

/* ---------- title ---------- */
.main-title {
    font-size: 42px; font-weight: 800; text-align: center; margin-bottom: 5px;
    background: linear-gradient(90deg, #ffffff, #38bdf8, #a78bfa, #ffffff);
    background-size: 300% 100%;
    -webkit-background-clip: text; background-clip: text; color: transparent;
    animation: shine 6s linear infinite, dropIn .9s ease both;
}
.main-title .bot { display: inline-block; -webkit-text-fill-color: initial; animation: wave 2.4s ease-in-out infinite; transform-origin: 70% 70%; }
@keyframes shine  { to { background-position: 300% 0; } }
@keyframes dropIn { from { opacity: 0; transform: translateY(-30px); } to { opacity: 1; transform: none; } }
@keyframes wave   { 0%,60%,100% { transform: rotate(0); } 10%,30% { transform: rotate(14deg); } 20%,40% { transform: rotate(-10deg); } 50% { transform: rotate(8deg); } }

.subtitle {
    text-align: center; color: #cbd5e1; font-size: 18px; margin-bottom: 30px;
    animation: fadeUp 1s ease .3s both;
}
@keyframes fadeUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: none; } }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background-color: #111827; }
[data-testid="stSidebar"] [data-testid="stAlert"] { animation: slideIn .6s ease both; transition: transform .25s; }
[data-testid="stSidebar"] [data-testid="stAlert"]:hover { transform: translateX(6px); }
@keyframes slideIn { from { opacity: 0; transform: translateX(-30px); } to { opacity: 1; transform: none; } }

/* ---------- feature cards ---------- */
.feature-card {
    background-color: #1e293b; padding: 20px; border-radius: 15px; border: 1px solid #334155;
    text-align: center; color: white; position: relative; overflow: hidden;
    animation: cardIn .8s cubic-bezier(.2,.8,.2,1) both;
    transition: transform .3s, box-shadow .3s, border-color .3s;
}
.feature-card:hover { transform: translateY(-10px) scale(1.03); border-color: #38bdf8; box-shadow: 0 14px 34px rgba(56,189,248,.35); }
.feature-card::after {           /* light sweep */
    content: ""; position: absolute; top: 0; left: -120%; width: 60%; height: 100%;
    background: linear-gradient(120deg, transparent, rgba(255,255,255,.14), transparent);
    animation: sweep 5s ease-in-out infinite;
}
.feature-card .fi { font-size: 34px; display: inline-block; animation: floaty 3s ease-in-out infinite; }
.c1 { animation-delay: .1s; } .c2 { animation-delay: .25s; } .c3 { animation-delay: .4s; } .c4 { animation-delay: .55s; }
.c2::after { animation-delay: 1s; } .c3::after { animation-delay: 2s; } .c4::after { animation-delay: 3s; }
.c2 .fi { animation-delay: .4s; } .c3 .fi { animation-delay: .8s; } .c4 .fi { animation-delay: 1.2s; }
@keyframes cardIn { from { opacity: 0; transform: translateY(40px) scale(.92); } to { opacity: 1; transform: none; } }
@keyframes floaty { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
@keyframes sweep  { 0%,60% { left: -120%; } 100% { left: 160%; } }

/* ---------- chat ---------- */
.stChatMessage, [data-testid="stChatMessage"] {
    border-radius: 15px; padding: 10px;
    background: rgba(30,41,59,.65); border: 1px solid #334155; margin-bottom: 10px;
    animation: pop .5s cubic-bezier(.2,.9,.3,1.2) both;
}
@keyframes pop { from { opacity: 0; transform: translateY(18px) scale(.94); } to { opacity: 1; transform: none; } }

/* thinking dots */
.thinking { display: flex; align-items: center; gap: 10px; color: #cbd5e1; }
.thinking .d { width: 10px; height: 10px; border-radius: 50%; background: #38bdf8; animation: bounce 1.2s infinite ease-in-out; }
.thinking .d:nth-child(2) { animation-delay: .15s; background: #818cf8; }
.thinking .d:nth-child(3) { animation-delay: .3s; background: #a78bfa; }
@keyframes bounce { 0%,80%,100% { transform: translateY(0); opacity: .4; } 40% { transform: translateY(-10px); opacity: 1; } }

/* tool badges */
.tb { display: inline-block; margin: 4px 6px 4px 0; padding: 6px 14px; border-radius: 999px; font-size: 14px; color: #fff;
      background: linear-gradient(90deg, #6366f1, #0ea5e9); animation: pop .4s both, glow 1.4s ease-in-out infinite; }
.tb.done { background: #16a34a; animation: pop .4s both; }
@keyframes glow { 0%,100% { box-shadow: 0 0 0 rgba(56,189,248,0); } 50% { box-shadow: 0 0 16px rgba(56,189,248,.8); } }

.cursor { display: inline-block; width: 8px; height: 16px; background: #38bdf8; margin-left: 2px; vertical-align: middle; animation: blink .8s infinite; }
@keyframes blink { 50% { opacity: 0; } }

/* ---------- buttons + input ---------- */
.stButton button { width: 100%; border-radius: 10px; font-weight: bold; transition: transform .2s, box-shadow .2s; }
.stButton button:hover { transform: translateY(-3px); box-shadow: 0 8px 22px rgba(99,102,241,.45); }
[data-testid="stChatInput"] { transition: box-shadow .3s; border-radius: 14px; }
[data-testid="stChatInput"]:focus-within { box-shadow: 0 0 0 2px #38bdf8, 0 0 26px rgba(56,189,248,.45); }

</style>

<div class="bubbles"><span></span><span></span><span></span><span></span><span></span><span></span></div>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================

st.markdown('<div class="main-title"><span class="bot">🤖</span> LangGraph Multi AI Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An Intelligent AI Assistant Powered by LangGraph & Tools</div>', unsafe_allow_html=True)

# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.title("⚙️ Multi AI Agent")
    st.divider()
    st.subheader("🔧 Available Tools")
    st.success("🔎 Web Search")
    st.success("🧮 Calculator")
    st.success("📈 Stock Price")
    st.success("🌤️ Weather")
    st.divider()

    st.subheader("✨ Try asking")
    for q in ["What is 89898 * 9099999909876 + 7342312323?", "Weather in Chennai?", "TSLA stock price", "Latest news on LangGraph"]:
        if st.button(q, key=q):
            st.session_state.pending = q

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Powered by LangGraph 🚀")

# =====================================================
# TOOLS
# =====================================================

search_tool = TavilySearch(max_results=5, topic="general", search_depth="advanced")


@tool
def calculator(expression: str) -> str:
    """
    Useful for mathematical calculations.

    Examples:
    2 + 2
    10 * 5
    math.sqrt(16)
    """
    try:
        allowed = {"math": math, "abs": abs, "round": round, "min": min, "max": max, "sum": sum}
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"Calculation Error: {str(e)}"


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Get latest stock price.

    Example:
    AAPL
    TSLA
    MSFT
    """
    api_key = os.getenv("ALPHA_API_KEY")
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    )
    return requests.get(url, timeout=10).json()


@tool
def get_current_weather(location: str) -> str:
    """
    Get current weather information.

    Example:
    Chennai
    London
    New York
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Weather API key is missing. Please configure OPENWEATHER_API_KEY."

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={location}&appid={api_key}&units=metric"
    )
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if response.status_code != 200:
            return f"Weather Error: {data.get('message')}"
        return (
            f"🌍 Location: {location}\n\n"
            f"🌡️ Temperature: {data['main']['temp']} °C\n\n"
            f"☁️ Condition: {data['weather'][0]['description'].title()}\n\n"
            f"💧 Humidity: {data['main']['humidity']}%"
        )
    except Exception as e:
        return f"Weather Error: {str(e)}"


tools = [search_tool, calculator, get_stock_price, get_current_weather]

TOOL_LABEL = {
    "tavily_search": "🔎 Web Search",
    "calculator": "🧮 Calculator",
    "get_stock_price": "📈 Stock Price",
    "get_current_weather": "🌤️ Weather",
}

# =====================================================
# LANGGRAPH
# =====================================================


@st.cache_resource
def create_agent():
    llm = ChatGroq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))
    llm_with_tools = llm.bind_tools(tools)

    class ChatState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    def chat_node(state: ChatState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph = StateGraph(ChatState)
    graph.add_node("Chat Node", chat_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "Chat Node")
    graph.add_conditional_edges("Chat Node", tools_condition)
    graph.add_edge("tools", "Chat Node")
    return graph.compile()


chatbot = create_agent()

# =====================================================
# FEATURE CARDS
# =====================================================

col1, col2, col3, col4 = st.columns(4)

cards = [
    (col1, "c1", "🔎", "Web Search", "Search information online"),
    (col2, "c2", "🧮", "Calculator", "Solve mathematical problems"),
    (col3, "c3", "📈", "Stock Market", "Get stock information"),
    (col4, "c4", "🌤️", "Weather", "Get live weather updates"),
]
for col, cls, icon, name, desc in cards:
    with col:
        st.markdown(
            f'<div class="feature-card {cls}"><span class="fi">{icon}</span><br><b>{name}</b><br>{desc}</div>',
            unsafe_allow_html=True,
        )

st.write("")
st.divider()

# =====================================================
# CHAT HISTORY
# =====================================================

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage) and message.content and not message.tool_calls:
        with st.chat_message("assistant"):
            st.write(message.content)

# =====================================================
# CHAT INPUT
# =====================================================

user_input = st.chat_input("Ask me anything...")
if st.session_state.pending:
    user_input, st.session_state.pending = st.session_state.pending, None

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        status = st.empty()
        answer = st.empty()

        status.markdown(
            '<div class="thinking"><div class="d"></div><div class="d"></div><div class="d"></div>'
            "<span>AI is thinking...</span></div>",
            unsafe_allow_html=True,
        )

        try:
            badges: list[tuple[str, bool]] = []
            new_msgs: list[BaseMessage] = []
            final_text = ""

            def show_badges():
                html_badges = "".join(
                    f'<span class="tb {"done" if done else ""}">{"✅ " if done else "⚙️ "}{label}</span>'
                    for label, done in badges
                )
                status.markdown(html_badges, unsafe_allow_html=True)

            for update in chatbot.stream({"messages": st.session_state.messages}, stream_mode="updates"):
                for _, payload in update.items():
                    for msg in payload.get("messages", []):
                        new_msgs.append(msg)

                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for tc in msg.tool_calls:
                                badges.append((TOOL_LABEL.get(tc["name"], tc["name"]), False))
                            show_badges()

                        elif isinstance(msg, ToolMessage):
                            label = TOOL_LABEL.get(msg.name, msg.name)
                            for i, (lb, done) in enumerate(badges):
                                if lb == label and not done:
                                    badges[i] = (lb, True)
                                    break
                            show_badges()
                            time.sleep(0.3)

                        elif isinstance(msg, AIMessage):
                            final_text = msg.content if isinstance(msg.content, str) else str(msg.content)

            # typewriter effect for the final answer
            words = final_text.split(" ")
            step = max(1, len(words) // 80)
            shown = ""
            for i in range(0, len(words), step):
                shown = " ".join(words[: i + step])
                answer.markdown(shown + '<span class="cursor"></span>', unsafe_allow_html=True)
                time.sleep(0.03)
            answer.markdown(final_text)

            st.session_state.messages.extend(new_msgs)

        except Exception as e:
            status.empty()
            st.error(f"Error: {str(e)}")