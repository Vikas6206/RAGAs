import pytest
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics.collections import AnswerRelevancy, FactualCorrectness
from utils import get_llm_response, load_test_data
from ragas.embeddings.base import embedding_factory


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "getData",
    load_test_data("testdata.json"),
    indirect=True,
)
async def test_relevancy_factual(llm_wrapper, getData):

    print("Testing relevance evaluation framework :::::::: ")

    embeddings = embedding_factory(
        "openai",
        model="text-embedding-3-small",
        client=llm_wrapper.client
    )

    metrics = [
        AnswerRelevancy(llm=llm_wrapper, embeddings=embeddings),
        FactualCorrectness(llm=llm_wrapper, embeddings=embeddings)
    ]

    eval_dataset = EvaluationDataset([getData])

    results = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=llm_wrapper,
        embeddings=embeddings
    )

    print(results)

    assert results["answer_relevancy"].value > 0.7
    assert results["factual_correctness"].value > 0.7


@pytest.fixture
def getData(request):
    test_data = request.param
    responseDict = get_llm_response(test_data)
    # get all the data set and convert it to the format of single turn sample which is expected by the evaluation framework
    sample = SingleTurnSample (
        user_input= test_data["question"], 
        response= responseDict["answer"],
        retrieved_contexts= [ doc["page_content"]for doc in responseDict.get("retrieved_docs", [])], # looping over all the response to get page_content
        reference = test_data["reference"]
    )
    return sample    