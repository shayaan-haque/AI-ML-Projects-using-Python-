from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain.tools.render import render_text_description
from dotenv import load_dotenv
load_dotenv()

@tool
def get_length_of_string(string: str) -> int:
    """Will give the length of the string (excluding spaces)"""
    print(f"Getting the length of: {string}")
    stripped_text = string.replace(" ","")
    print(f"Stripped text: {stripped_text}")
    return len(stripped_text)

if __name__ == "__main__":
    llm = ChatGoogleGenerativeAI(
        temprature=0,
        model = "gemini-1.0-pro",
        max_tokens=1024,
    )

    tools = [get_length_of_string]
    template = """
    Answer the follwing questions the best you can.
    You have access to the follwing tools {tools}
    Use the following format:

    Question: The input question you must answer
    Thought: you should always think about what to do
    Action: the action to take , should be one of [{tool_names}]
    Action Input: the input to the action (just the value,no function call syntax)
    Observation: the result of the Action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!
    Question:{input}
    Thought:"""

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=",".join([t.name for t in tools])
    )

    chain = {"input": lambda x:x["input"]} | prompt | llm
    
    res = chain.invoke(
        {"input":"what is the length in characters of text 'Mac Book'?"}
    )

    print(res)
