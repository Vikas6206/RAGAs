#Pytest - 
import os

from openai import AsyncOpenAI
import pytest
from ragas.metrics.collections import ContextRecall
import requests

# #user_input ->query
# #response -> response
# #reference -> Ground truth
# #retrived_context -> Top K retrieved docs

@pytest.mark.asyncio
async def test_context_precision(llm_wrapper, getData):
    question = "How many articles are there in the Selenium webdriver python course?"
      #create object of class of metric
    context_recall = ContextRecall(llm=llm_wrapper)

    responseDict = getData
    print("Executing test for question: ", question)
    print("Response keys:", responseDict.keys())  # Debug: see what keys exist

    # Evaluate
    result = await context_recall.ascore(
        user_input=question,
        retrieved_contexts=[responseDict["retrieved_docs"][0]["page_content"],
                            responseDict["retrieved_docs"][1]["page_content"],
                            responseDict["retrieved_docs"][2]["page_content"]],
        reference="23"
    )
#     #Get the score 
    print(result.value)
    assert result.value > 0.7

 

@pytest.fixture
def getData():
    # Question is the actual data being feed as an input to the test here
    question = "How many articles are there in the Selenium webdriver python course?"
    responseDict = requests.post("https://rahulshettyacademy.com/rag-llm/ask", json={
        "question":question,
        "chat_history":[]
    }).json()
    return responseDict