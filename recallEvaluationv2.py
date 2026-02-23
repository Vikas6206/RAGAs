#Pytest - 
import os

from openai import AsyncOpenAI
import pytest
from ragas.metrics.collections import ContextRecall
from ragas.llms import llm_factory
import requests

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
    client = AsyncOpenAI()
    llm_openAI = llm_factory(
        model="gpt-4o-mini",
        client=client,
        temperature=0
    )

    # llm_langChain = LangchainLLMWrapper(llm=llm_openAI)
#     #create object of class of metric
    context_recall = ContextRecall(llm=llm_openAI)
#     # Feed data - singleTurnSample (only one conversation for testing)

    responseDict = requests.post("https://rahulshettyacademy.com/rag-llm/ask", json={
        "question":question,
        "chat_history":[]
    }).json()


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