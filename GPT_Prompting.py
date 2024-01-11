import openai
import os
import re
from dotenv import load_dotenv


class PromptingGPT:

    # Load API key and organization from environment variables
    load_dotenv("secrets.env")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organization = os.getenv("OPENAI_ORGANIZATION")

    ClientOpenAi = openai.OpenAI(
            api_key= openai.api_key,
            organization= openai.organization
        )

    conversation_history = []
    def make_thread_prompt(self, prompt):
        """Combine previous messages with the current prompt and creating a thread
        with gpt-3.5-turbo-1106"""
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
        for msg in self.conversation_history:
            messages.append({'role': 'user', 'content': msg})

        messages.append({'role': 'user', 'content': prompt})

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
            max_tokens=4000,
        )

        # Extract and print the model's reply
        reply = response.choices[0].message.content
        print(reply)

        # Update conversation history
        self.conversation_history.append(prompt)
        self.conversation_history.append(reply)
        return reply

    def make_prompts_turbo(self, prompt):
        """
        Combine previous messages with the current prompt and creating a thread
        with gpt-3.5-turbo-1106
        """
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]

        messages.append({'role': 'user', 'content': prompt})

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
            max_tokens=4000,
            temperature=0.9,
        )

        # Extract and print the model's reply
        reply = response.choices[0].message.content
        return reply

    def make_prompt_GPT_instruct(self, prompt):
        """Combine previous messages with the current prompt and creating a thread
        with gpt-3.5-instruct"""

        response = openai.completions.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=3950,
            temperature=0.7,  # 0 to 1 by  step of 0.1 O deterministic, 1 is very creative
        )

        # Extract and return the model's reply
        reply = response.choices[0].text
        return reply

    def chat_prompts(self):
        """Creation of a chat with GPT 3.5 turbo for chatting and fast Q&A"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        state = True
        while state:
            pat_close = re.compile(
                r'(Bye|goodnight|ok thank you)', flags=re.IGNORECASE)
            message = input("You: ")
            if message:
                messages.append(
                    {"role": "user", "content": message},
                )
                chat_completion = openai.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    messages=messages
                )
            answer = chat_completion.choices[0].message.content
            print(f"ChatGPT: {answer}")
            messages.append({"role": "assistant", "content": answer})
            if re.search(pat_close, message):
                state = False
