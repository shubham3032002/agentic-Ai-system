import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Web search agent
websearch_agent = Agent(
    name="Web Search Agent",
    role="Search the web for relevant information.",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources and ensure that results are relevant to the user's query."],
    show_tool_calls=True,
    markdown=True,
)

# Financial Agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True
        )
    ],
    instructions=["Use tables to display financial data. Always include sources, and focus on analyst recommendations and company news for the given stock."],
    show_tool_calls=True,
    markdown=True,
)

# Multi-Agent System
multi_ai_agent = Agent(
    model=Groq(id='llama3-groq-70b-8192-tool-use-preview'),
    team=[websearch_agent, finance_agent],
    instructions=[
        "If the query is related to financial data (stock price, analyst recommendations), use the Finance AI Agent.",
        "If the query is related to general news or web search, use the Web Search Agent.",
        "Always include sources in the response.",
        "Use tables for financial data when applicable."
    ],
    show_tool_calls=True,
    markdown=True
)

# Streamlit UI
st.title("Multi-Agent Query System")

# Query input
user_query = st.text_input("Enter your query:")

if st.button("Submit"):
    if user_query:
        with st.spinner("Processing your query..."):
            try:
                # Use print_response instead of execute
                response = multi_ai_agent.print_response(user_query, stream=True)
                
                # Debugging: print response to terminal
                print(response)
                
                # Check if response is valid and not empty
                if response:
                    st.write("### Response:")
                    st.markdown(response)
                else:
                    st.warning("No response received. Please check the query or agent setup.")
            except Exception as e:
                st.error(f"Error processing the query: {e}")
    else:
        st.warning("Please enter a query.")
