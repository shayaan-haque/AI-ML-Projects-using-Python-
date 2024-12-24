from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain 
from dotenv import load_dotenv
load_dotenv()

memory = ConversationBufferWindowMemory(k=2)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
conversation = ConversationChain(llm = llm , memory = memory ,verbose=True)

while True:
    user_input = input("\n Shayaan:")

    if user_input.lower() in ['bye','exit']:
        print("Bye")
    
        print(conversation.memory.buffer)
        break

    response = conversation.predict(input = user_input)
    print("\n AI: ", response)
    
