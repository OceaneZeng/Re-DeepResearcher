from __future__ import annotations

from langchain.agents import AgentExecutor, create_react_agent

from .llm import get_llm
from .prompts import get_react_prompt
from .tools import calculator, list_dir, read_text_file


def build_react_agent(*, verbose: bool = True) -> AgentExecutor:
    """Build a ReAct agent executor with a small toolset."""
    llm = get_llm()
    tools = [calculator, list_dir, read_text_file]
    prompt = get_react_prompt()
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
    )

