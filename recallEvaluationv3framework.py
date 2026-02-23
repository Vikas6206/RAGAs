import pytest
from ragas.metrics.collections import ContextRecall
from utils import get_llm_response, load_test_data


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "getData",
    load_test_data("recallEvaluationv3framework.json"),
    indirect=True,   # ⭐ IMPORTANT FIX
)
async def test_context_precision(llm_wrapper, getData):

    context_recall = ContextRecall(llm=llm_wrapper)
    # responseDict = getData  # Now this is API response
    result = await context_recall.ascore(
        user_input= getData["question"],
        retrieved_contexts= [
            getData["retrieved_docs"][0]["page_content"],
            getData["retrieved_docs"][1]["page_content"],
            getData["retrieved_docs"][2]["page_content"],
        ],
        reference=getData["reference"],
    )

    print(result.value)
    assert result.value > 0.7


@pytest.fixture
def getData(request):
    test_data = request.param
    responseDict = get_llm_response(test_data)
    # attach question back into response since it can't be accessed directly on the test function
    responseDict["question"] = test_data["question"]
    responseDict["reference"] = test_data["reference"]
    return responseDict