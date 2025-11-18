import asyncio
import os
import time

from dotenv import load_dotenv

# os.chdir("codif-ape-graph-rag/src/")
from llm.client import sync_get_llm_client

load_dotenv()
client = sync_get_llm_client()

names = [
    "Alice",
    "Bob",
    "Charlie",
    "David",
    "Emma",
    "Frank",
    "Grace",
    "Hannah",
    "Ian",
    "Jessica",
    "Kevin",
    "Linda",
    "Michael",
    "Nancy",
    "Olivia",
    "Peter",
    "Quincy",
    "Rachel",
    "Samuel",
    "Tiffany",
]

messages_batch = [
    [
        {
            "role": "user",
            "content": f"Say hello to the new user on Paradigm! Give a one sentence short highly personalized welcome to: {name}",
        }
    ]
    for name in names
]


async def send_message(messages, *args, **kwargs):
    response = await client.chat.completions.create(messages=messages, *args, **kwargs)
    return response


async def main():
    start_time = time.time()
    tasks = [send_message(messages, model=os.environ["GENERATION_MODEL"], temperature=0.4) for messages in messages_batch]
    responses = await asyncio.gather(*tasks)
    duration = time.time() - start_time
    print(f"Async execution took {duration} seconds.")
    for response in responses:
        print(response.choices[0].message.content)
    return responses


if __name__ == "__main__":
    asyncio.run(main())
