import pytest
from ragas.metrics.collections import Faithfulness
from utils import get_llm_response, load_test_data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "getData",
    load_test_data("faithFullnessEvaluationframework.json"),
    indirect=True,   # ⭐ IMPORTANT FIX
)
async def test_faithfulness(llm_wrapper, getData):
    print("Tesing failthfulness evaluation framework :::::::: ")
    faithfulness = Faithfulness(llm=llm_wrapper)
    # responseDict = getData  # Now this is API response
    responseDict = getData

    print("Response from LLM application  :::::::::::::::::::: ", responseDict)
    result = await faithfulness.ascore(
        user_input= responseDict["question"], 
        response= responseDict["answer"],
        retrieved_contexts= [ doc["page_content"]for doc in responseDict.get("retrieved_docs", [])] # looping over all the response to get page_content
    )
    print(result.value)
    assert result.value > 0.7


@pytest.fixture
def getData(request):
    test_data = request.param
    responseDict = get_llm_response(test_data)
    # attach question back into response since it can't be accessed directly on the test function
    responseDict["question"] = test_data["question"]
    # responseDict["reference"] = test_data["reference"]
    return responseDict