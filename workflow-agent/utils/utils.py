

import logging
import os
import time

from langsmith import Client
import langsmith

def post_sharable_url(issue, langsmith_run_id, time_delay_s):
  logging.info(f"🚀 Posting sharable URL for LangSmith run: {langsmith_run_id}")
  sharable_url = get_langsmith_trace_sharable_url(langsmith_run_id, time_delay_s=time_delay_s)
  text = f"👉 [Follow the bot's progress in real time on LangSmith]({sharable_url})."
  logging.info(f"Sharable URL: {text}")

def get_langsmit_run_from_metadata(metadata_value, metadata_key="run_id_in_metadata") -> langsmith.schemas.Run:
  """This will only return the FIRST match on single metadta field

  Args:
      metadata_key (str, optional): _description_. Defaults to "run_id_in_metadata".
      metadata_value (str, optional): _description_. Defaults to "b187061b-afd7-40ab-a918-705cf16219c3".

  Returns:
      Run: _description_
  """
  langsmith_client = Client()
  runs = langsmith_client.list_runs(project_name=os.environ['LANGCHAIN_PROJECT'])

  count = 0
  for _r in runs:
    count += 1
  print(f"Found num runs: {count}")

  for run in langsmith_client.list_runs(project_name=os.environ['LANGCHAIN_PROJECT']):
    if run.extra and run.extra.get('metadata') and run.extra.get('metadata').get(metadata_key) == metadata_value:
      # return the 'top-level' of the trace (keep getting runs' parents until at top)
      if run.parent_run_id:
        curr_run = run
        while curr_run.parent_run_id:
          curr_run = langsmith_client.read_run(str(curr_run.parent_run_id))
        return curr_run
      else:
        return run

def get_langsmith_trace_sharable_url(run_id_in_metadata, project_name='', time_delay_s=0):
  """

  Adding metadata to runs: https://docs.smith.langchain.com/tracing/tracing-faq#how-do-i-add-metadata-to-runs
  
  Background: 
    A 'Trace' is a collection of runs organized in a tree or graph. The 'Root Run' is the top level run in a trace.
  https://docs.smith.langchain.com/tracing/tracing-faq

  Args:
      project (_type_): _description_
  """
  time.sleep(time_delay_s)
  if project_name == '':
    project_name = os.environ['LANGCHAIN_PROJECT']

  langsmith_client = Client()

  # re-attempt to find the run, maybe it hasn't started yet.
  run = None
  for _i in range(8):
    run = get_langsmit_run_from_metadata(str(run_id_in_metadata), metadata_key="run_id_in_metadata")
    if run is not None:
      break
    print(f"Attempt {_i} to find run with metadata {run_id_in_metadata}")
    time.sleep(5)

  if run is None:
    return f"Failed to generate sharable URL, cannot find this run on LangSmith. RunID: {run_id_in_metadata}"

  if not langsmith_client.run_is_shared(run.id):
    sharable_url = langsmith_client.share_run(run_id=run.id)
  else:
    sharable_url = langsmith_client.read_run_shared_link(run_id=run.id)
  logging.info(f'⭐️ sharable_url: {sharable_url}')
  return sharable_url
