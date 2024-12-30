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

@tool
def convert_to_uppercase(string: str) -> str:
    """Will convert the string to uppercase"""
    print(f"Converting to uppercase: {string}")
    uppercased_string = string.upper()
    print(f"Uppercased string: {uppercased_string}")
    return uppercased_string

@tool
def count_vowels(string: str) -> int:
    """Will count the number of vowels in the string"""
    vowels = "aeiouAEIOU"
    vowel_count = sum(1 for char in string if char in vowels)
    print(f"Vowel count in '{string}': {vowel_count}")
    return vowel_count

if __name__ == "__main__":

    llm = ChatGoogleGenerativeAI(
        temprature=0,
        model="gemini-1.0-pro",
        max_tokens=1024,
    )

    tools = [get_length_of_string, convert_to_uppercase, count_vowels]

   
    template = """
    Answer the following questions the best you can.
    You have access to the following tools {tools}
    Use the following format:

    Question: The input question you must answer
    Thought: You should always think about what to do
    Action: The action to take, should be one of [{tool_names}]
    Action Input: The input to the action (just the value, no function call syntax)
    Observation: The result of the Action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: The final answer to the original input question

    Begin!
    Question: {input}
    Thought:"""


    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=",".join([t.name for t in tools])
    )


    chain = {"input": lambda x: x["input"]} | prompt | llm
    
    res = chain.invoke(
        {"input": "What is the length in characters of text 'Mac Book', how many vowels are there, and convert it into upper case?"}
    )

    print(res)
