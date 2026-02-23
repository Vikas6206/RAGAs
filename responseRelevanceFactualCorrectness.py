import pytest
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics.collections import AnswerRelevancy ,FactualCorrectness
from utils import get_llm_response, load_test_data
from ragas.embeddings.base import embedding_factory


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "getData",
    load_test_data("testdata.json"),
    indirect=True,   # ⭐ IMPORTANT FIX
)
async def test_relevancy_factual(llm_wrapper,getData):
    print("Tesing relevance evaluation framework :::::::: ")
    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=llm_wrapper.client)
    metrics = [AnswerRelevancy(llm=llm_wrapper, embeddings=embeddings), FactualCorrectness(llm=llm_wrapper)]
     # responseDict = getData  # Now this is API response
    eval_dataset = EvaluationDataset([getData]) # convert singleton sample to raga data set format
    results = evaluate(dataset=eval_dataset, metrics=metrics)  # this will run the evaluation for all the metrics in the list and print the results

   

    print(results)




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