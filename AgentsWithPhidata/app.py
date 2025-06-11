from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
import openai
from phi.tools.yfinance import YFinanceTools
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

web_search_agent = Agent(
    name = "web search",
    role = "Search the web for latest information",
    model = Groq(id="deepseek-r1-distill-llama-70b"),
    tools = [DuckDuckGo()],
    instructions = ["Always include sources"],
    show_tool_calls = True,
    markdown = True
)

finance_agent = Agent(
    name = "Finance Agent",
    role = "Gather information about stock data",
    model = Groq(id="deepseek-r1-distill-llama-70b"),
    tools = [YFinanceTools(stock_price = True, analyst_recommendations = True, stock_fundamentals = True, company_news = True)],
    instructions = ["use tables to display the data"],
    show_tool_calls = True,
    markdown = True
)

multi_agent = Agent(
    team = [web_search_agent, finance_agent],
    name = "Multi Agent",
    instruction = ["Always include sources", "Always use tables to display the data"],
    markdown = True,
    show_tool_calls = True
)

multi_agent.print_response("Summarize analyst recommendation and share the latest news for NVDA",stream = True)