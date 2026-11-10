from email import message

from dotenv import load_dotenv

load_dotenv()

import ollama

from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3.5:2b"


#tools

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """lookup the price of a product in the catalog"""
    print(f"    >>>Executing get_product_price(product='{product}')")
    prices = {"laptop": 999, "headphones": 45, "keyboard": 30}
    return prices.get(product,0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """"apply a discount tier to a price and return the price.
    Aavailable tiers: gold, silver, bronze"""
    print(f"    >>>Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_precentages = {"gold":20, "silver":10, "bronze":5}
    discount = discount_precentages.get(discount_tier,0)
    return round(price * (1-discount/100), 2)

#without @tool decorator we must manually define the json schema for each function
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "look up price of product in the catalog",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "the product name eg. laptop, phone, keyboard"
                    }
                },
                "required": ["product"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


# NOTE: Ollama can also auto-generate these schemas if you pass the functions
# directly as tools (similar to LangChain's @tool decorator):
#   tools_for_llm = [get_product_price, apply_discount]
# However, this requires your docstrings to follow the Google docstring format
# so Ollama can parse parameter descriptions from the Args section. For example:
#   def get_product_price(product: str) -> float:
#       """Look up the price of a product in the catalog.
#
#       Args:
#           product: The product name, e.g. 'laptop', 'headphones', 'keyboard'.
#
#       Returns:
#           The price of the product, or 0 if not found.
#       """
# We keep the manual JSON version here so you can see what @tool hides from you.

# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.

@traceable(name="ollama_chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)




#agent loop
@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"question = {question}")
    print("=" * 60)

    messages: list = [
        {
            "role": "system",
            "content": (
                "you are a helpful shopping assistant you have access to a product catalog tool and a discount tool. you do not need product specifics. you can look for the products with the product catalog tool \n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price"
                "you must call get_product_price first to get the real price\n"
                "2. only call apply_discount after you have recieved a price from get_product_price. pass the exact price returned by get_product_price - do NOT pass a made up number\n"
                "3. never calculate discounts yourself using math. ALWAYS use the apply_discount tool\n"
                "4. if the user doesn't specify a discount tier ask them which tier to use - do NOT assume a tier"
            ),
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n----------ITERATION {iteration}----------")

        response = ollama_chat_traced(messages=messages)

        ai_message = response.message

        tool_calls = ai_message.tool_calls

        #if no tool calls, this is final answer
        if not tool_calls:
            print(f"\nfinal answer: {ai_message.content}")
            return ai_message.content

        #process only the first tool call - force one tool per iteration 
        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"tool selected : {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"tool: '{tool_name}' not found")

        observation = tool_to_use(**tool_args)
        print(f"    [tool results] {observation}")

        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )

    print("ERROR: max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")