# 🚀 Day 6: Publish Your Agent

Welcome to day six of our AI Agents Crash Course.

We have done a lot of things so far:

* Created a pipeline for processing any GitHub repository  
* Ingested this data into a search engine  
* Created an agent that uses the search engine as a tool  
* Evaluated this agent

But this agent still lives inside our Jupyter Notebook. It's time to make it available for everyone

That's what we will do today:

* Clean the code  
* Create a UI for it with Streamlit  
* Deploy it to the Internet

At the end of this lesson, you'll be able to share the link to your agent on social media, so anyone can interact with it.

# Cleaning up

Maybe while following the course you kept your code organized, but in my case, I have everything in one Jupyter Notebook. By now it's quite messy.

So the first step is to organize everything into multiple clean Python files. This makes our code easier to maintain, test, and deploy.

First, let's create a separate folder called "app".

Inside I initialized an empty uv project and updated dependencies in `pyproject.toml`:

```
dependencies = [
    "minsearch>=0.0.5",
    "openai>=1.108.2",
    "pydantic-ai==1.0.9",
    "python-frontmatter>=1.1.0",
    "requests>=2.32.5",
]
```

I didn't include sentence transformers but feel free to add it if you plan to use vector search.

Now install the dependencies:

```
uv sync
```

I created the following files:

* ingest.py \- handles data loading and indexing from GitHub repositories  
* search\_tools.py \- contains our search tool implementation  
* search\_agent.py \- creates and configures the Pydantic AI agent  
* logs.py \- handles logging of conversations  
* main.py \- brings everything together for the command-line interface

Let me include the code for each here.

**ingest.py**

```py
import io
import zipfile
import requests
import frontmatter

from minsearch import Index


def read_repo_data(repo_owner, repo_name):
    url = f'https://codeload.github.com/{repo_owner}/{repo_name}/zip/refs/heads/main'
    resp = requests.get(url)

    repository_data = []

    zf = zipfile.ZipFile(io.BytesIO(resp.content))

    for file_info in zf.infolist():
        filename = file_info.filename.lower()

        if not (filename.endswith('.md') or filename.endswith('.mdx')):
            continue

        with zf.open(file_info) as f_in:
            content = f_in.read()
            post = frontmatter.loads(content)
            data = post.to_dict()

            _, filename_repo = file_info.filename.split('/', maxsplit=1)
            data['filename'] = filename_repo
            repository_data.append(data)

    zf.close()

    return repository_data


def sliding_window(seq, size, step):
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    result = []
    for i in range(0, n, step):
        batch = seq[i:i+size]
        result.append({'start': i, 'content': batch})
        if i + size > n:
            break

    return result


def chunk_documents(docs, size=2000, step=1000):
    chunks = []

    for doc in docs:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop('content')
        doc_chunks = sliding_window(doc_content, size=size, step=step)
        for chunk in doc_chunks:
            chunk.update(doc_copy)
        chunks.extend(doc_chunks)

    return chunks


def index_data(
        repo_owner,
        repo_name,
        filter=None,
        chunk=False,
        chunking_params=None,
    ):
    docs = read_repo_data(repo_owner, repo_name)

    if filter is not None:
        docs = [doc for doc in docs if filter(doc)]

    if chunk:
        if chunking_params is None:
            chunking_params = {'size': 2000, 'step': 1000}
        docs = chunk_documents(docs, **chunking_params)

    index = Index(
        text_fields=["content", "filename"],
    )

    index.fit(docs)
    return index
```

I made a few improvements here. The line `_, filename_repo = file_info.filename.split('/', maxsplit=1)` in `read_repo_data` strips the first part of the path (the zip archive name), making it easier for our agent to create references.

I also replaced `chunk` with `content` in `sliding_window`, so the content is always in the `content` field, and our code works with or without chunking.

Finally, we have the `index_data` function that combines all the ingestion steps. This covers what we did in days 1-3.

Days 4 and 5 are in `search_agent.py` and `search_tools.py`.

**search\_tools.py:**

```py
from typing import List, Any

class SearchTool:
    def __init__(self, index):
        self.index = index

    def search(self, query: str) -> List[Any]:
        """
        Perform a text-based search on the FAQ index.

        Args:
            query (str): The search query string.

        Returns:
            List[Any]: A list of up to 5 search results returned by the FAQ index.
        """
        return self.index.search(query, num_results=5)
```

I created a class instead of just a function like we had in the Jupyter notebook. Previously, it was a global variable that we referenced from a function. Now the `index` is encapsulated inside a class with tools, which makes the code more organized.

**search\_agent.py**

```py
import search_tools
from pydantic_ai import Agent


SYSTEM_PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions about documentation.  

Use the search tool to find relevant information from the course materials before answering questions.  

If you can find specific information through search, use it to provide accurate answers.

Always include references by citing the filename of the source material you used.
Replace it with the full path to the GitHub repository:
"https://github.com/{repo_owner}/{repo_name}/blob/main/"
Format: [LINK TITLE](FULL_GITHUB_LINK)


If the search doesn't return relevant results, let the user know and provide general guidance.
"""

def init_agent(index, repo_owner, repo_name):
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(repo_owner=repo_owner, repo_name=repo_name)

    search_tool = search_tools.SearchTool(index=index)

    agent = Agent(
        name="gh_agent",
        instructions=system_prompt,
        tools=[search_tool.search],
        model='gpt-4o-mini'
    )

    return agent
```

Here we use a template instead of hardcoding the repository information, so it's more flexible and can work with any code repository.

Then we have another change: `tools=[search_tool.search]` instead of the previous `tools=[text_search]` in `Agent` because the tool we want to use is now a method of the `search_tool` class.

All the materials from day 5 (yesterday) are in **logs.py:**

```py
import os
import json
import secrets
from pathlib import Path
from datetime import datetime

from pydantic_ai.messages import ModelMessagesTypeAdapter


LOG_DIR = Path(os.getenv('LOGS_DIRECTORY', 'logs'))
LOG_DIR.mkdir(exist_ok=True)


def log_entry(agent, messages, source="user"):
    tools = []

    for ts in agent.toolsets:
        tools.extend(ts.tools.keys())

    dict_messages = ModelMessagesTypeAdapter.dump_python(messages)

    return {
        "agent_name": agent.name,
        "system_prompt": agent._instructions,
        "provider": agent.model.system,
        "model": agent.model.model_name,
        "tools": tools,
        "messages": dict_messages,
        "source": source
    }


def serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def log_interaction_to_file(agent, messages, source='user'):
    entry = log_entry(agent, messages, source)

    ts = entry['messages'][-1]['timestamp']
    ts_str = ts.strftime("%Y%m%d_%H%M%S")
    rand_hex = secrets.token_hex(3)

    filename = f"{agent.name}_{ts_str}_{rand_hex}.json"
    filepath = LOG_DIR / filename

    with filepath.open("w", encoding="utf-8") as f_out:
        json.dump(entry, f_out, indent=2, default=serializer)

    return filepath
```

The main change here is `LOG_DIR = Path(os.getenv('LOGS_DIRECTORY', 'logs'))`. This allows us to configure the log directory using an environment variable. It's useful for deployment and makes our code more flexible.

Finally, **main.py** puts everything together:

```py
import ingest
import search_agent 
import logs

import asyncio


REPO_OWNER = "DataTalksClub"
REPO_NAME = "faq"


def initialize_index():
    print(f"Starting AI FAQ Assistant for {REPO_OWNER}/{REPO_NAME}")
    print("Initializing data ingestion...")

    def filter(doc):
        return 'data-engineering' in doc['filename']

    index = ingest.index_data(REPO_OWNER, REPO_NAME, filter=filter)
    print("Data indexing completed successfully!")
    return index


def initialize_agent(index):
    print("Initializing search agent...")
    agent = search_agent.init_agent(index, REPO_OWNER, REPO_NAME)
    print("Agent initialized successfully!")
    return agent


def main():
    index = initialize_index()
    agent = initialize_agent(index)
    print("\nReady to answer your questions!")
    print("Type 'stop' to exit the program.\n")

    while True:
        question = input("Your question: ")
        if question.strip().lower() == 'stop':
            print("Goodbye!")
            break

        print("Processing your question...")
        response = asyncio.run(agent.run(user_prompt=question))
        logs.log_interaction_to_file(agent, response.new_messages())

        print("\nResponse:\n", response.output)
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
```

This creates a simple command-line interface for our agent. We use `asyncio.run()` because Pydantic AI's `run` method is asynchronous.

Of course, this code isn't perfect.

We don't have documentation or tests. If you're using Cursor or GitHub Copilot, you can easily add these. I'd also create a CI/CD pipeline that runs tests every time you push code to GitHub. Finally, the logs should be ideally saved to a proper storage service like S3.

But we won't do that here. Let's focus on getting our agent online.

# Streamlit

The code is clean and modular, so we can now use it to create a UI. We'll use Streamlit.

Streamlit is a Python library that makes it easy to create web applications. You write Python code, and it automatically creates an interactive web interface.

Let's install it:

```shell
uv add streamlit
```

I don't know much about Streamlit, and can only create simple interfaces. But luckily, we have many AI coding assistants that can help us.

I'll use ChatGPT today and share the results with you. But you can experiment with your favorite tool too. If you don't have one yet, try ChatGPT. Starting from GPT-4, ChatGPT is quite good at coding tasks. You can also try GitHub Copilot or Cursor.

Pydantic AI has a nice gallery with examples. Among other things, I saw [Chat App with FastAPI](https://ai.pydantic.dev/examples/chat-app/#example-code). We can use it as inspiration with minimal changes.

I'll use it for creating a Streamlit app.

My prompt for ChatGPT:

```
I have this agent I created with Pydantic AI

[insert main.py]

I want to turn it into streamlit code. Base it on the following code for creating web apps with Pydantic AI:

[insert example from docs]
```

You can see my conversation [here](https://chatgpt.com/share/68d7ac70-0ccc-800a-bdfa-c82cd0a3744e).

I got some code and put it into app.py:

```py
import streamlit as st
import asyncio

import ingest
import search_agent
import logs


# --- Initialization ---
@st.cache_resource
def init_agent():
    repo_owner = "DataTalksClub"
    repo_name = "faq"

    def filter(doc):
        return 'data-engineering' in doc['filename']

    st.write("🔄 Indexing repo...")
    index = ingest.index_data(repo_owner, repo_name, filter=filter)
    agent = search_agent.init_agent(index, repo_owner, repo_name)
    return agent


agent = init_agent()

# --- Streamlit UI ---
st.set_page_config(page_title="AI FAQ Assistant", page_icon="🤖", layout="centered")
st.title("🤖 AI FAQ Assistant")
st.caption("Ask me anything about the DataTalksClub/faq repository")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask your question..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = asyncio.run(agent.run(user_prompt=prompt))
            answer = response.output
            st.markdown(answer)

    # Save response to history + logs
    st.session_state.messages.append({"role": "assistant", "content": answer})
    logs.log_interaction_to_file(agent, response.new_messages())
```

Let's run it:

```shell
uv run streamlit run app.py
```

This is what I got:

![][image1]

It worked on the first attempt\!

Note that we don't keep conversation history between questions. When you ask the next question, the agent doesn't remember previous exchanges.

We can fix this by passing \`message\_history\` in the \`run\` method of the agent. You can see how I implemented this in [ToyAIKit's PydanticAIRunner class](https://github.com/alexeygrigorev/toyaikit/blob/main/toyaikit/chat/runners.py#L259).

I played with it and thought: "It would be nice if responses streamed instead of displaying everything at once". Pydantic AI has [an example with streaming](https://ai.pydantic.dev/examples/stream-markdown/), so I asked ChatGPT to adjust our app.py.

My prompt:

```
Can you make it streaming? Here's an example from PydanticAI:

[example]
```

After a few back-and-forth exchanges, I got this code:

```py
import streamlit as st
import asyncio

import ingest
import search_agent
import logs


# --- Initialization ---
@st.cache_resource
def init_agent():
    repo_owner = "DataTalksClub"
    repo_name = "faq"

    def filter(doc):
        return "data-engineering" in doc["filename"]

    st.write("🔄 Indexing repo...")
    index = ingest.index_data(repo_owner, repo_name, filter=filter)
    agent = search_agent.init_agent(index, repo_owner, repo_name)
    return agent


agent = init_agent()

# --- Streamlit UI ---
st.set_page_config(page_title="AI FAQ Assistant", page_icon="🤖", layout="centered")
st.title("🤖 AI FAQ Assistant")
st.caption("Ask me anything about the DataTalksClub/faq repository")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# --- Streaming helper ---
def stream_response(prompt: str):
    async def agen():
        async with agent.run_stream(user_prompt=prompt) as result:
            last_len = 0
            full_text = ""
            async for chunk in result.stream_output(debounce_by=0.01):
                # stream only the delta
                new_text = chunk[last_len:]
                last_len = len(chunk)
                full_text = chunk
                if new_text:
                    yield new_text
            # log once complete
            logs.log_interaction_to_file(agent, result.new_messages())
            st.session_state._last_response = full_text

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    agen_obj = agen()

    try:
        while True:
            piece = loop.run_until_complete(agen_obj.__anext__())
            yield piece
    except StopAsyncIteration:
        return


# --- Chat input ---
if prompt := st.chat_input("Ask your question..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant message (streamed)
    with st.chat_message("assistant"):
        response_text = st.write_stream(stream_response(prompt))

    # Save full response to history
    final_text = getattr(st.session_state, "_last_response", response_text)
    st.session_state.messages.append({"role": "assistant", "content": final_text})
```

I don't want to understand all the details of this code (if I wanted to, I'd ask ChatGPT to explain it). But hey \- it works\! 

You could even ask ChatGPT to display tool calls and other debugging information. Feel free to experiment with that.

But for now, let's deploy what we already have.

# Deployment

Streamlit Cloud [should understand uv](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies#other-python-package-managers), but it didn't work for me. So I exported the dependencies into requirements.txt:

```shell
uv export --no-dev > requirements.txt
```

Push all your code to GitHub \- that's how Streamlit Cloud will access it.

By now you probably noticed the "Deploy" button at the top right corner. Let's press it.

Congratulations\! Your application is deployed.

It won't work immediately though, because we need to provide it with an OpenAI API key. Go to [https://share.streamlit.io/](https://share.streamlit.io/), find your app there, and add your key in the secrets:

```
OPENAI_API_KEY="your-key"
```

If you want to be more careful with your API keys, you can create a project in OpenAI (e.g., "streamlit") and use a project-specific key. This way you can easily monitor your spending for this particular project.

That's it\! Our app is deployed and working.

Here's my app: [https://aiherodefaq.streamlit.app/](https://aiherodefaq.streamlit.app/) 

By the time you read this, it'll probably be disabled. Every time somebody uses it, I have to pay OpenAI. And I don't want that.

But tomorrow we'll wrap everything up and record a video about it for publishing on social media\!

# Homework

* Clean up your agent code into modular Python files  
* Create a Streamlit interface for your agent  
* Deploy your app to Streamlit Cloud and share it (optionally)  
* Share your progress on social media

## Example post for LinkedIn

| 🚀 Day 6 of building AI agent: publishing the agent\! Today I took my agent from Jupyter notebook to the Internet: • Cleaned up messy code into modular Python files • Built a web interface with Streamlit • Deployed to the cloud with one click • Now anyone can interact with my agent\! My project helps users search through \[YOUR PROJECT DESCRIPTION\]. The agent can answer questions and provide relevant links from the documentation. Here's my live app: \[YOUR\_APP\_LINK\] My repo: \[YOUR\_REPO\_LINK\] Tomorrow: final touches and course wrap-up\! Following along with this amazing course \- who else is building AI agents? You can sign up here: [https://alexeygrigorev.com/aihero/](https://alexeygrigorev.com/aihero/) |
| :---- |

## Example post for Twitter/X

| Day 6: My AI agent is LIVE\! 🚀 ✅ Cleaned up messy notebook code ✅ Built Streamlit interface ✅ Deployed to cloud in minutes ✅ Anyone can now use my agent From local notebook → production web app Here's my live app: \[YOUR\_APP\_LINK\] Here's my repo: \[YOUR\_REPO\_LINK\] Next: Course wrap-up & final demo Join me: https://alexeygrigorev.com/aihero/ |
| :---- |

# Community

Have questions about this lesson or suggestions for improvement? You can find me and other learners in **DataTalks.Club Slack**:

* [Join DataTalks.Club](https://datatalks.club/slack.html)  
* Find us in the [`#course-ai-bootcamp` channel](https://app.slack.com/client/T01ATQK62F8/C09DLTMKVHV)

In the community channel, you can:

* Ask questions about the course content  
* Share your implementation and get feedback  
* Show off your deployed applications  
* Get help with deployment issues  
* Connect with other course participants

Don't hesitate to reach out \- the community is here to help each other succeed\!  


[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA8sAAAGkCAIAAACJriBeAACAAElEQVR4Xuy9DZwVxZ33++xz7+597j57z64mBkVNfEt82dx49wmsupP1JagxYiJifEFcFTQiKCISECIgMaAoIOqoE5mRKEomwCpiYDUSfAlLMCSgkskIhllFhmA8xIlz4qjnPvfu86/6d9eprn91nz4zPcyZw68+X3FOd3VVdb3+uvpf1f8ld8AAAAAAAAAAQFb8F3kIAAAAAAAA0G2gsAEAAAAAAMgSKGwAAAAAAACyBAobAAAAAACALIHCBgAAAAAAIEugsAEAAAAAAMgSKGwAAAAAAACyBAobAAAAAACALIHCBgAAAAAAIEugsAEAAAAAAMgSKGwAAAAAAACyBAobAAAAAACALIHCBgAAAAAAIEugsAEAAAAAAMgSKGwAAAAAAACyBAobAAAAAACALIHCBgAAAAAAIEugsAEAAAAAAMgSKGwAAAAAAACyBAobAAAAAACALIHCBgAAAAAAIEugsAEAAAAAAMgSKGwAAAAAAACyBAobAAAAAACALIHCBgAAAAAAIEugsAEAAAAAAMgSKGwAAAAAAACyBAobAAAAAACALIHCBqBm2b79zU8++aQYuo8++ujMs4dKbwAAAADIFihsAGqQwSd95bXXXjfa2jgS3NIzAAAAALIFChuAGmTGbbd3FgquvtbuwotGSP8AAAAAyBAo7PLs3bv3/Ri3fMWT0r+Xrw89/7etb7jXv//+nj3vzpz1fek/DW5YlTsZpuGMIee8vvU37gWhe+utt+UlFXHkMcdv+MXGjo4OVwCG7qOPPnpk8aMn150ur82We+693709y6Uv4qri+bU/czM0dHPuuEv6z5DFjy5Z8viPbBY1PiK91QB33b3AudNbp98mvQEAANgPgcIuj6tQLLdjRxspUXmJ5Bvnf2vnznfc64vFP//5w24rHjesyp0M0zBx0pQ//elP7gWh+/jjj+Ul6fniiYP+8If33EB9jvJnxMgrZQgZsv7fN7ixWo6KWF5S/Ty96ifunYSu2/UtJVRkTox//OMfpbcaYOtvWpw7pQcb6Q0AAMB+SFUr7J/97IWEOU7jPvroI5JBvTTZOW/+Qje+qPvxshXyKkm/U9i/bX3D9R11F196ubyqLGeePdRee5fe3TJ1ugwtEz780FWEjpOXVD8XXjRiV3u7eyfaHXnM8dJ/hrjx1ajCvvyK0e+++wfnTqGwAQAAMFWqsEkEvPXW287olexIqr7wwksyqB6S8Lad3RvbtsmrJP1OYZMqcn1H3YKF98urkhl9zXVvv73TDSido4x65IePyTB7jhuTcINP+oq8qvp5bMlS906Kxffey0uf2eJGWaMK+5prx8k2AoUNAACAqTqFTdq67JxisiMNJ4PtHgnLxYxLuTlD/1LYVAquV+Hoduim5LVxjJ8wyQ2ickeqUYbcE6iI3TiE6yVlvw8gFbjxlV/yXZAcXPNvz0k/2XLGkHOimRdELX32dx586OGPP/7YuVMobAAAAEx1KeyT607/xcZXnEGL3a729ldffc2sKPrxshWtb2yLsze4ffadMvBukGDMars0r937l8K+/IrRrlfhKkr5hReN6Pbste3ogWfuXfNl+N0mTRG/vvU38kLgheq5m301qrAfWfyoe59Q2AAAAEKqSGGTvHYU84svvpxmldsXTxw0/5777AvJffDBB9Jnpcilft71eQ8+9LC81qF/KWy5+K+rq8s5UqxkFWDZ9xKFwp957440lvdZ2dx7V3NSzXGO9HBZ537FrdNvc3KvWKMKe9OmX7v3CYUNAAAgpIoU9hNLm+2x6s03fyf9JND84+XOJgY93/fXDo3dT1b/m3uoWKSxVl7rsM8UdstvW50dxBKQYRIXX3p5++7dTrDvvvsHKbI/6OyUl0u885rGSQN6el56Z9cu15/lHmpYJGPpBj9etsINWhvWu4fiH0WAw51z57l5V6MKmxqae59Q2AAAAEKqRWHb35/75JNPfvr8WnOKhPKy5f/60UcfWQNZ4LZsefWuuxfY4bz3Xt6cJekmI0rPmLE3WFEp9/777183brwUmnREXu6wzxR2z8d47+LOdete9ErPsqsAEza1oIJe+7N18hJm0uSp7gWh+/jjjx997Al5SaXIlWpUxAsW3i+LuBvLOonLLh/1619v+f3v93xArrOTAqefjz/hf7CJ44snDrr5O7e0vrFt7969hcKfOzo6KJyXXl5/X/2D0nNW2CmnYqIY6W9K+airx0jPNl6rm24obLrrpT/6MdUcipraPt04Ne2tv2m58abvpDHKSglVTropytjOQoEylp7qn1r5tPTmRW4kUsyi9QEAAKgNqkJhk0qzR6krR32bj69e82xZ6wJ27bt3m9BWPbPa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>