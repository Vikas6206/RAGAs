#Pytest - 
import os

from langchain_openai import ChatOpenAI
import pytest
from ragas import SingleTurnSample
from ragas.metrics.collections import LLMContextRecall
from ragas.llms import LangchainLLMWrapper
from requests import request

# #user_input ->query
# #response -> response
# #reference -> Ground truth
# #retrived_context -> Top K retrieved docs

@pytest.mark.asyncio
async def test_context_precision():

#     #power of LLM + method metric ->score
    os.environ["OPENAI_API_KEY"] = "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"
    question = "How many articles are there in the Selenium webdriver python course?"
#     #llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    llm_openAI = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    llm_langChain = LangchainLLMWrapper(llm=llm_openAI)
#     #create object of class of metric
    context_recall = LLMContextRecall(llm=llm_langChain)
#     # Feed data - singleTurnSample (only one conversation for testing)

    responseDict = request.post("https://rahulshettyacademy.com/rag-llm/ask", json={
        "question":question,
        "chat_history":[]
    }).json()

    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=[responseDict["retrieved_docs"][0]["page_content"],
                            responseDict["retrieved_docs"][1]["page_content"],
                            responseDict["retrieved_docs"][2]["page_content"]],
        reference="23"
    )

#     #Get the score 
#     # single_turn_metric is a async method which will not wait for the entire result to be processed and hence need await statement
    score = await context_recall.single_turn_metric(sample)
    print(score)
    assert score > 0.7