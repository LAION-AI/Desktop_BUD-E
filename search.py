import os
import subprocess
import time
import re
from langchain_core.prompts import ChatPromptTemplate
#from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from buddy import extract_urls_to_open

# Load system prompt from file
with open('system_prompt.txt', 'r') as file:
    system_prompt = file.read().strip()

# Define LanguageModelProcessor class
class LanguageModelProcessor:
    def __init__(self):
        # Initialize the language model (LLM)
        # self.llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=os.getenv("GROQ_API_KEY"))
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=os.getenv("OPENAI_API_KEY"))

        # Initialize conversation memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Create chat prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        # Create conversation chain
        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )

    def process(self, text):
        # Add user message to memory
        self.memory.chat_memory.add_user_message(text)

        # Record start time
        start_time = time.time()

        # Get response from LLM
        response = self.conversation.invoke({"text": text})
        
        # Record end time
        end_time = time.time()

        # Add AI response to memory
        self.memory.chat_memory.add_ai_message(response['text'])

        # Calculate elapsed time
        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM ({elapsed_time}ms): {response['text']}")
        return response['text']

    def clear_chat_history(self):
        # Clear the conversation memory
        self.memory.chat_memory.clear()
        print("Chat history cleared.")

def main():
    llm_processor = LanguageModelProcessor()
    
    # Hardcoded user request for a Google search
    user_request = "Search for the latest advancements in AI technology on Google."
    
    # Process the user request
    llm_response = llm_processor.process(user_request)
    
    # Extract the Google search URL from the LLM response
    urls = extract_urls_to_open(llm_response)

    def fetch_html(url):
        try:
            result = subprocess.run(['curl', '-s', url], capture_output=True, check=True)
            return result.stdout.decode('utf-8', errors='replace')
        except subprocess.CalledProcessError as e:
            print(f"Error fetching the URL: {e}")
            return None

    def extract_top_links(html_content):
        prompt = f"Extract the top 5 links from the following HTML content and tag them with <open-url>:</open-url> with no extraneous information in the reply:\n\n{html_content}"
        response = llm_processor.process(prompt)
        return extract_urls_to_open(response)

    def fetch_individual_summaries(top_links):
        def summarize_html_in_chunks(html_content):
            chunk_size = 100000  # 100 KB
            chunks = [html_content[i:i + chunk_size] for i in range(0, len(html_content), chunk_size)]
            ongoing_summary = ""
            for i, chunk in enumerate(chunks):
                llm_processor.clear_chat_history()  # Clear chat history before sending new chunk
                if i == 0:
                    summary_prompt = f"Provide a detailed and nuanced summary of the content of the following HTML chunk, ignoring anything related to the UI, marketing, or technical details unrelated to the primary content intended for consumption:\n\n{chunk}"
                else:
                    summary_prompt = f"Refine the ongoing summary with detailed and nuanced context from the previous summary and the current HTML chunk, ignoring anything related to the UI, marketing, or technical details unrelated to the primary content intended for consumption:\n\n{ongoing_summary}\n\nCurrent HTML chunk:\n\n{chunk}"
                print(f"Sending HTML chunk {i+1} to LLM for summarization.")
                ongoing_summary = llm_processor.process(summary_prompt)
            return ongoing_summary

        individual_summaries = []
        for link in top_links:
            print(f"Fetching HTML content for link: {link}")
            html_content = fetch_html(link)
            if html_content:
                print(f"Breaking HTML content of {link} into chunks for summarization.")
                summary_response = summarize_html_in_chunks(html_content)
                llm_processor.clear_chat_history()  # Clear chat history after summarizing individual content
                individual_summaries.append(summary_response)
            else:
                print(f"Failed to fetch HTML content for link: {link}")
        return individual_summaries

    def summarize_combined_content(individual_summaries):
        delimiter = "\n---\n"
        combined_summaries = delimiter.join(individual_summaries)
        final_summary_prompt = f"Provide a detailed and nuanced summary of the content of the following summaries, each separated by '{delimiter}':\n\n{combined_summaries}"
        print("Sending combined summaries to LLM for final summarization.")
        final_summary_response = llm_processor.process(final_summary_prompt)
        return final_summary_response

    def extract_and_summarize_html_content(urls):
        for url in urls:
            print(f"Google Search URL: {url}")
            html_content = fetch_html(url)
            if html_content:
                print("Extracting the Top 5 links for the search results")
                top_links = extract_top_links(html_content)
                print(f"Top 5 links from the search results:\n{top_links}")

                individual_summaries = fetch_individual_summaries(top_links)
                final_summary_response = summarize_combined_content(individual_summaries)
                print(f"Final summary of the top 5 links:\n{final_summary_response}")
            else:
                print(f"Failed to fetch HTML content for URL: {url}")

    if urls:
        extract_and_summarize_html_content(urls)
    else:
        print("No Google search URL found in the response.")

if __name__ == "__main__":
    main()
