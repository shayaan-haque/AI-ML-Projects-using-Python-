from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
if __name__ == "__main__":
    to = input("enter the language you want to convert it in")
    sentence = input("enter the sentence")
    prompt = """
    "{sentence} change this sentence to {language}
    """
    # output_parser = StrOutputParser()
    prompt_template = PromptTemplate(input_variables=["language","sentence"],template=prompt)
    llm = ChatGoogleGenerativeAI(
        model='gemini-1.5-flash',
    )
    chain = prompt_template | llm | StrOutputParser()
    res = chain.invoke({"language":to,"sentence":sentence})
    print(res)

