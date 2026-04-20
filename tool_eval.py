from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import exact
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, system_message, use_tools
from inspect_ai.tool import Tool, tool


@tool
def get_weather() -> Tool:
    async def execute(city: str) -> str:
        """Get the current weather for a city.

        Args:
            city (str): The name of the city to get weather for.
        """
        weather_data = {
            "san francisco": "foggy, 58F",
            "new york": "sunny, 72F",
            "seattle": "rainy, 52F",
            "chicago": "windy, 65F",
            "los angeles": "sunny, 78F",
        }
        return weather_data.get(city.lower(), f"Weather data not available for {city}")
    return execute


@tool
def get_population() -> Tool:
    async def execute(city: str) -> str:
        """Get the population of a city.

        Args:
            city (str): The name of the city to get population for.
        """
        population_data = {
            "san francisco": "873,965",
            "new york": "8,336,817",
            "seattle": "749,256",
            "chicago": "2,696,555",
            "los angeles": "3,898,747",
        }
        return population_data.get(city.lower(), f"Population data not available for {city}")
    return execute


@tool
def calculator() -> Tool:
    async def execute(expression: str) -> str:
        """Evaluate a basic math expression.

        Args:
            expression (str): A math expression to evaluate like 2 + 2 or 347 * 12.
        """
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    return execute


dataset = [
    Sample(input="What is the weather in San Francisco?", target="foggy, 58F"),
    Sample(input="What is the population of New York?", target="8,336,817"),
    Sample(input="What is 347 multiplied by 12?", target="4164"),
    Sample(input="What is the weather in Seattle?", target="rainy, 52F"),
    Sample(input="What is 89 + 213?", target="302"),
]


@task
def tool_use_eval():
    return Task(
        dataset=dataset,
        solver=[
            system_message(
                "You are a helpful assistant with access to tools. "
                "Use the tools to answer questions accurately. "
                "Answer with ONLY the exact value, no other words. For example, if asked about weather, reply only with the weather string like 'foggy, 58F'."
            ),
            use_tools([get_weather(), get_population(), calculator()]),
            generate(),
        ],
        scorer=exact(),
    )