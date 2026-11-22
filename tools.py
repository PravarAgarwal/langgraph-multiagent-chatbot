from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import math
from dotenv import load_dotenv

load_dotenv()

search_tool = TavilySearch(
    max_results = 5,
    topic = 'general',
    search_depth = 'advanced'
)

@tool
def calculator(expression: str) -> str:
    """
    useful for simple calculations.
    Input should be a valid math expression
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """

    try:
        allowed = {
            "math" : math,
            "abs" : abs,
            "round" : round,
            "min" : min,
            "max" : max,
            "sum" : sum

        }

        result = eval(expression, {"__builtins__", {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"