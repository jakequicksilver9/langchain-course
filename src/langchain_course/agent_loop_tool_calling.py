from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3.5:2b"


#tools

@tool
def get_product_price(product: str) -> float:
    """lookup the price of a product in the catalog"""
    print(f"    >>>Executing get_product_price(product='{product}')")
    prices = {"laptop": 999, "headphones": 45, "keyboard": 30}
    return prices.get(product,0)

@tool 
def apply_discount(price: float, discount_tier: str) -> float:
    """"apply a discount tier to a price and return the price.
    Aavailable tiers: gold, silver, bronze"""
    print(f"    >>>Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_precentages = {"gold":20, "silver":10, "bronze":5}
    discount = discount_precentages.get(discount_tier,0)
    return round(price * (1-discount/100), 2)

#agent loop
@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"question = {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content = (
                "you are a helpful shopping assistant you have access to a product catalog tool and a discount tool. you do not need product specifics. you can look for the products with the product catalog tool \n\n"
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price"
                "you must call get_product_price first to get the real price\n"
                "2. only call apply_discount after you have recieved a price from get_product_price. pass the exact price returned by get_product_price - do NOT pass a made up number\n"
                "3. never calculate discounts yourself using math. ALWAYS use the apply_discount tool\n"
                "4. if the user doesn't specify a discount tier ask them which tier to use - do NOT assume a tier"
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n----------ITERATION {iteration}----------")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        #if no tool calls, this is final answer
        if not tool_calls:
            print(f"\nfinal answer: {ai_message.content}")
            return ai_message.content

        #process only the first tool call - force one tool per iteration 
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"tool selected : {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"tool: '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)
        print(f"    [tool results] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id = tool_call_id)
        )

    print("ERROR: max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")