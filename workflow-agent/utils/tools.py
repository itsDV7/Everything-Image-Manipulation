import os
from typing import List

import langchain

from dotenv import load_dotenv
from langchain.agents import load_tools
from langchain.agents.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.tools import ShellTool
from langchain.agents.agent_toolkits.file_management.toolkit import FileManagementToolkit
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_community.tools import VectorStoreQATool
from langchain.tools import (BaseTool, StructuredTool)
from langchain.agents.agent_toolkits.playwright.toolkit import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import (
    create_async_playwright_browser,
    create_sync_playwright_browser,
)
from langchain_community.utilities.github import GitHubAPIWrapper

from utils.vector_db import get_vectorstore_retriever_tool

load_dotenv(override=True, dotenv_path='../../.env')
# This import is required only for jupyter notebooks, since they have their own eventloop
# import nest_asyncio

# nest_asyncio.apply()

os.environ["LANGCHAIN_TRACING"] = "true"  # If you want to trace the execution of the program, set to "true"
langchain.debug = False  # type: ignore
VERBOSE = True
root_dir = os.getenv("root_dir","/app")


def get_tools(langsmith_run_id: str, sync=True):
  """Main function to assemble tools for ML for Bio project."""

  # CODE EXECUTION - langsmith_run_id as unique identifier for the sandbox
  
  # SHELL & FILES
  shell = ShellTool()
  file_management = FileManagementToolkit(
    # If you don't provide a root_dir, operations will default to the current working directory
    root_dir=root_dir
  ).get_tools()

  # WEB BROWSER
  browser_toolkit = None
  if sync:
    sync_browser = create_sync_playwright_browser()
    browser_toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
  else:
    # TODO async is work in progress... not functional yet.
    async_browser = create_async_playwright_browser()
    browser_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
  browser_tools = browser_toolkit.get_tools()

  # HUMAN
  if os.environ['OPENAI_API_TYPE'] == 'azure':
    llm = AzureChatOpenAI(
        temperature=0.1,
        model="gpt-4-1106-Preview",

    )
    # max_retries=3,
    # request_timeout=60 * 3,
    # deployment_name=os.environ['AZURE_OPENAI_ENGINE'])  # type: ignore
  else:
    llm = ChatOpenAI(temperature=0.1, model="gpt-4-0613", max_retries=3, request_timeout=60 * 3)  # type: ignore
  human_tools = load_tools(["human"], llm=llm, input_func=get_human_input)
  # GOOGLE SEARCH
  search = load_tools(["serpapi"])

  # # GITHUB
  # github = GitHubAPIWrapper()  # type: ignore
  # toolkit = GitHubToolkit.from_github_api_wrapper(github)
  # github_tools: list[BaseTool] = toolkit.get_tools()

  # TODO: more vector stores per Bio package: trimmomatic, gffread, samtools, salmon, DESeq2 and ggpubr
  docs_tools: List[VectorStoreQATool] = [
      get_vectorstore_retriever_tool(
          course_name='langchain-docs',
          name='Langchain-docs',
          description=
          "Build context-aware, reasoning applications with LangChain's flexible abstractions and AI-first toolkit."),
      get_vectorstore_retriever_tool(
          course_name='ml4bio-star',
          name='STAR-docs',
          description=
          'Basic STAR workflow consists of 2 steps: (1) Generating genome indexes files and (2) Mapping reads to the genome'
      ),
      get_vectorstore_retriever_tool(
          course_name='ml4bio-fastqc',
          name='FastQC-docs',
          description=
          'FastQC aims to provide a simple way to do some quality control checks on raw sequence data coming from high throughput sequencing pipelines. It provides a modular set of analyses which you can use to give a quick impression of whether your data has any problems of which you should be aware before doing any further analysis. It works with data from BAM, SAM or FastQ files'
      ),
      get_vectorstore_retriever_tool(
          course_name='ml4bio-multiqc',
          name='MultiQC-docs',
          description=
          "MultiQC is a reporting tool that parses results and statistics from bioinformatics tool outputs, such as log files and console outputs. It helps to summarize experiments containing multiple samples and multiple analysis steps. It's designed to be placed at the end of pipelines or to be run manually when you've finished running your tools."
      ),
      get_vectorstore_retriever_tool(
          course_name='ml4bio-bioconductor',
          name='Bioconductor-docs',
          description=
          "Bioconductor is a project that contains hundreds of individual R packages. They're all high quality libraries that provide widespread access to a broad range of powerful statistical and graphical methods for the analysis of genomic data. Some of them also facilitate the inclusion of biological metadata in the analysis of genomic data, e.g. literature data from PubMed, annotation data from Entrez genes."
      ),
  ]

  # ARXIV SEARCH
  # Probably unnecessary: WikipediaQueryRun, WolframAlphaQueryRun, PubmedQueryRun, ArxivQueryRun
  # arxiv_tool = ArxivQueryRun()

  tools: list[BaseTool] =  search + docs_tools + [shell] + browser_tools + file_management + human_tools # + github_tools
  return tools


############# HELPERS ################
# def _should_check(serialized_obj: dict) -> bool:
#   # Only require approval on ShellTool.
#   return serialized_obj.get("name") == "terminal"

# def _approve(_input: str) -> bool:
#   if _input == "echo 'Hello World'":
#     return True
#   msg = ("Do you approve of the following input? "
#          "Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no.")
#   msg += "\n\n" + _input + "\n"
#   resp = input(msg)
#   return resp.lower() in ("yes", "y")


def get_human_input() -> str:
  """Placeholder for Slack/GH-Comment input from user."""
  print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
  contents = []
  while True:
    try:
      line = input()
    except EOFError:
      break
    if line == "q":
      break
    contents.append(line)
  return "\n".join(contents)


if __name__ == "__main__":
  tools = get_tools(sync=True, langsmith_run_id="MY RUN ID FROM OUTSIDE")
  # print(tools)
  # print("SCHEMA: ", tools.args_schema.schema_json(indent=2))
  if type(tools) == List:
    # raise Exception("No tools found.")
    pass
  else:
    tools[0].run("print('Hello World from inside the tools.run() function!')")
