from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain 
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit = 200
)
conversation = ConversationChain(llm = llm , memory = memory,verbose = True)

while True:
    user_input = input("\n Shayaan:")

    if user_input.lower() in ['bye','exit']:
        print("Bye")
    
        print("\nConversation Summary:")
        print(conversation.memory.buffer)
        break

    response = conversation.predict(input = user_input)
    print("\n AI: ", response)
    