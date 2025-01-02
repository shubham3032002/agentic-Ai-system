from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv
load_dotenv()
#web search agent
openai.api_key=os.getenv("OPEN_AI_API_KEY")
websearch_agent=Agent(
    name="web search agent",
    role='Search the web for infomation',
    model=Groq(id='llama3-groq-70b-8192-tool-use-preview'),
    tools=[
        DuckDuckGo()
    ],
    instructions=['Alway include sources'],
    show_tool_calls=True,
    markdown=True,
)


#Financial Agent 

finance_agent=Agent(
    name='Finance AI Agent',
    model=Groq(id='llama3-groq-70b-8192-tool-use-preview'),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
           company_news=True)],
    instructions=['use table to display the data'],
    show_tool_calls=True,
    markdown=True,  
)

multi_ai_agent=Agent(
    model=Groq(id='llama3-groq-70b-8192-tool-use-preview'),
    team=[websearch_agent,finance_agent],
    instructions=['Alway include sources,"use table to dislpay data'],
    show_tool_calls=True,
    markdown=True
)

multi_ai_agent.print_response('Summarize Analyst recommandation and share the latest new for NVDA',stream=True)