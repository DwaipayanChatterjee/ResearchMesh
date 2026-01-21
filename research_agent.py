# =========================
# Imports
# =========================
import streamlit as st
import os

from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.models.openai import OpenAIChat
from agno.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.hackernews import HackerNewsTools
from agno.tools.newspaper4k import Newspaper4kTools


# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="AI Researcher",
    page_icon="🤖",
    layout="wide",
)


# =========================
# Custom CSS
# =========================
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .sub-title {
            font-size: 1.1rem;
            color: #6b7280;
            margin-bottom: 2rem;
        }
        .result-box {
            padding: 1.5rem;
            border-radius: 12px;
            background-color: #0f172a;
            color: #e5e7eb;
        }
        .stTextInput > div > div > input {
            border-radius: 10px;
        }
        .stButton button {
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Header
# =========================
st.markdown('<div class="main-title">🔍 ResearchMesh - Multi-Agent AI Researcher</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Research HackerNews, read articles, search the web, and generate high-quality summaries, blogs, or reports.</div>',
    unsafe_allow_html=True,
)


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("⚙️ Configuration")

    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Your API key is used only for this session.",
    )

    st.markdown("---")

    st.markdown(
        """
        **🤖 Agents in this App**
        - HackerNews Researcher  
        - Web Search Agent  
        - Article Reader  

        **📝 Output**
        - Insightful summaries  
        - Blog-ready content  
        - Research reports
        """
    )


# =========================
# Guard Clause
# =========================
if not openai_api_key:
    st.info("👈 Please enter your OpenAI API key in the sidebar to get started.")
    st.stop()

os.environ["OPENAI_API_KEY"] = openai_api_key


# =========================
# Agents
# =========================
hn_researcher = Agent(
    name="HackerNews Researcher",
    model=OpenAIChat(id="gpt-4o-mini"),
    role="Finds and analyzes top HackerNews stories.",
    tools=[HackerNewsTools()],
)

web_searcher = Agent(
    name="Web Searcher",
    model=OpenAIChat(id="gpt-4o-mini"),
    role="Searches the web for deeper context.",
    tools=[DuckDuckGoTools()],
    add_datetime_to_context=True,
)

article_reader = Agent(
    name="Article Reader",
    model=OpenAIChat(id="gpt-4o-mini"),
    role="Reads and extracts insights from articles.",
    tools=[Newspaper4kTools()],
)

hackernews_team = Team(
    name="HackerNews AI Team",
    model=OpenAIChat(id="gpt-4o-mini"),
    members=[hn_researcher, web_searcher, article_reader],
    instructions=[
        "Search HackerNews for relevant stories.",
        "Read the linked articles for context.",
        "Search the web for additional insights.",
        "Produce a thoughtful, engaging summary.",
    ],
    markdown=True,
    debug_mode=False,
    show_members_responses=False,
)


# =========================
# Main Input Section
# =========================
st.markdown("### 🧠 What do you want to research?")

query = st.text_input(
    "Enter your topic or question",
    placeholder="e.g. Latest trends in Generative AI on HackerNews",
)

run_button = st.button("🚀 Run Research")


# =========================
# Run & Output
# =========================
if run_button and query:
    with st.spinner("🔎 Researching across agents..."):
        response: RunOutput = hackernews_team.run(query, stream=False)

    st.success("✅ Research complete!")

    st.markdown("### 📄 Research Output")
    st.markdown(
        f"""
        <div class="result-box">
            {response.content}
        </div>
        """,
        unsafe_allow_html=True,
    )

elif run_button and not query:
    st.warning("⚠️ Please enter a research query.")
