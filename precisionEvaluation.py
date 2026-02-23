#Pytest - 
import os

from openai import AsyncOpenAI
import pytest
from ragas import SingleTurnSample, evaluate
from ragas.metrics.collections.context_precision import ContextPrecisionWithoutReference
from ragas.llms import llm_factory

# #user_input ->query
# #response -> response
# #reference -> Ground truth
# #retrived_context -> Top K retrieved docs

@pytest.mark.asyncio
async def test_context_precision():

#     #power of LLM + method metric ->score
    os.environ["OPENAI_API_KEY"] = "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"

#     #llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    client = AsyncOpenAI()
    llm = llm_factory(
        model="gpt-4o-mini",
        client=client,
        temperature=0
    )
#     #create object of class of metric
    metric = ContextPrecisionWithoutReference(llm=llm)
#     # Feed data - singleTurnSample (only one conversation for testing)

    sample = SingleTurnSample(
        user_input="How many articles are there in the Selenium webdriver python course?",
        response="There are 23 articles in the course.",
        retrieved_contexts=["Complete Understanding on Selenium Python API Methods with real time Scenarios on LIVE Websites\n\"Last but not least\" you can clear any Interview and can Lead Entire Selenium Python Projects from Design Stage\nThis course includes:\n17.5 hours on-demand video\nAssignments\n23 articles\n9 downloadable resources\nAccess on mobile and TV\nCertificate of completion\nRequirements",
                           "What you'll learn\nAt the end of this course, You will get complete knowledge on Python Automation using Selenium WebDriver\nYou will be able to implement Python Test Automation Frameworks from Scratch with all latest Technlogies\nComplete Understanding of Python Basics with many practise Examples to gain a solid exposure\nYou will be learning Python Unit Test Frameworks like PyTest which will helpful for Unit and Integration Testing",
                           "Wish you all the Best! See you all in the course with above topics :)\n\nWho this course is for:\nAutomation Engineers\nSoftware Engineers\nManual testers\nSoftware developers",
                           "So what makes this course Unique in the Market?\nWe assume that students have no experience in automation / coding and start every topic from scratch and basics.\nExamples are taken from  REAL TIME HOSTED WEB APPLICATIONS  to understand how different components can be automated.\n  Topics includes: \nPython Basics\nPython Programming examples\nPython Data types\nPython OOPS Examples\nSelenium Locators\nSelenium Multi Browser Execution\nPython Selenium API Methods\nAdvanced Selenium User interactions"]
    )

#     #Get the score 
#     # single_turn_metric is a async method which will not wait for the entire result to be processed and hence need await statement
    score = await metric.ascore(sample)
    print(score)