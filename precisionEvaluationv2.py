from openai import AsyncOpenAI
import pytest
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecisionWithoutReference
import os


@pytest.mark.asyncio
async def test_context_precision():
    
  # Setup LLM
  os.environ["OPENAI_API_KEY"] = "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"

  client = AsyncOpenAI()
  llm = llm_factory("gpt-4o-mini", client=client)

  # Create metric
  scorer = ContextPrecisionWithoutReference(llm=llm)

  # Evaluate
  result = await scorer.ascore(
        user_input="How many articles are there in the Selenium webdriver python course?",
        response="There are 23 articles in the course.",  
        retrieved_contexts=["Complete Understanding on Selenium Python API Methods with real time Scenarios on LIVE Websites\n\"Last but not least\" you can clear any Interview and can Lead Entire Selenium Python Projects from Design Stage\nThis course includes:\n17.5 hours on-demand video\nAssignments\n23 articles\n9 downloadable resources\nAccess on mobile and TV\nCertificate of completion\nRequirements",
                            "What you'll learn\nAt the end of this course, You will get complete knowledge on Python Automation using Selenium WebDriver\nYou will be able to implement Python Test Automation Frameworks from Scratch with all latest Technlogies\nComplete Understanding of Python Basics with many practise Examples to gain a solid exposure\nYou will be learning Python Unit Test Frameworks like PyTest which will helpful for Unit and Integration Testing",
                            "Wish you all the Best! See you all in the course with above topics :)\n\nWho this course is for:\nAutomation Engineers\nSoftware Engineers\nManual testers\nSoftware developers",
                            "So what makes this course Unique in the Market?\nWe assume that students have no experience in automation / coding and start every topic from scratch and basics.\nExamples are taken from  REAL TIME HOSTED WEB APPLICATIONS  to understand how different components can be automated.\n  Topics includes: \nPython Basics\nPython Programming examples\nPython Data types\nPython OOPS Examples\nSelenium Locators\nSelenium Multi Browser Execution\nPython Selenium API Methods\nAdvanced Selenium User interactions"]
  )
  print(result.value)
  print(f"Context Precision Score:::::; {result.value}")