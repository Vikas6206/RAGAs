
import os

from openai import AsyncOpenAI
import pytest
from ragas.llms import LangchainLLMWrapper, llm_factory


# global file which can be access from any test file in the folder - conftest.py
# we can create multiple fixtures here and use them in any test file by just importing the name of the fixture function. 
# No need to import the whole file. Pytest will automatically detect the fixtures in this file and make them available for use in any test file in the same folder.
#  This is a special feature of pytest that allows for easy sharing of setup code across multiple test files without the need for explicit imports. 
# Just define the fixture in conftest.py and use it in any test file in the same folder by referencing its name as a parameter in the test function. 
# Pytest will automatically find and execute the fixture before running the test function, providing the necessary setup or data for the test. 
# This promotes code reusability and keeps the test files clean and focused on the actual tests rather than setup code.  

@pytest.fixture
def llm_wrapper():
    os.environ["OPENAI_API_KEY"] = "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"
    client = AsyncOpenAI()
    llm_openAI = llm_factory(
        model="gpt-4o-mini",
        client=client,
        temperature=0
    )
    return llm_openAI   

