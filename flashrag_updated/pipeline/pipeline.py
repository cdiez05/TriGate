from flashrag.evaluator import Evaluator
from flashrag.dataset.utils import split_dataset, merge_dataset
from flashrag.utils import get_retriever, get_generator, get_refiner, get_judger
from flashrag.prompt import PromptTemplate
import torch
import requests


api_azure_key = "8e8049c8315f419489b50d9d095fe671"
# api_azure_key = "7014612e1b05439dac516f6883ae34d5"

class BasicPipeline:
    """Base object of all pipelines. A pipeline includes the overall process of RAG.
    If you want to implement a pipeline, you should inherit this class.
    """

    def __init__(self, config, prompt_template = None):
        self.config = config
        self.device = config['device']
        self.retriever = None
        self.evaluator = Evaluator(config)
        self.save_retrieval_cache = config['save_retrieval_cache']
        if prompt_template is None:
            prompt_template = PromptTemplate(config)
        self.prompt_template = prompt_template

    def run(self, dataset):
        """The overall inference process of a RAG framework."""
        pass

    def evaluate(self, dataset, do_eval=True, pred_process_fun=None):
        """The evaluation process after finishing overall generation"""

        if pred_process_fun is not None:
            raw_pred = dataset.pred
            processed_pred = [pred_process_fun(pred) for pred in raw_pred]
            dataset.update_output('raw_pred',raw_pred)
            dataset.update_output('pred', processed_pred)

        if do_eval:
            # evaluate & save result
            eval_result = self.evaluator.evaluate(dataset)
            print(eval_result)

        # save retrieval cache
        if self.save_retrieval_cache:
            self.retriever._save_cache()

        return dataset


class SequentialPipeline(BasicPipeline):
    def __init__(self, config, prompt_template = None):
        """
        inference stage:
            query -> pre-retrieval -> retriever -> post-retrieval -> generator
        """

        super().__init__(config, prompt_template)
        self.retriever = get_retriever(config)
        self.generator = get_generator(config)
        # TODO: add rewriter module

        self.use_fid = config['use_fid']

        if config['refiner_name'] is not None:
            self.refiner = get_refiner(config)
        else:
            self.refiner = None

    def naive_run(self, dataset, do_eval=True, pred_process_fun=None):
        # direct generation without RAG
        input_prompts = [self.prompt_template.get_string(question=q) for q in dataset.question]
        dataset.update_output('prompt', input_prompts)

        pred_answer_list = self.generator.generate(input_prompts)
        dataset.update_output("pred",pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
        return dataset
    
    def naive_output(self, dataset):
        # direct generation without RAG
        input_prompts = [self.prompt_template.get_string(question=q) for q in dataset.question]
        dataset.update_output('prompt', input_prompts)

        pred_answer_list = self.generator.generate(input_prompts)
        return pred_answer_list
    

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        input_query = dataset.question

        # print(self.retriever.batch_search(input_query))
        retrieval_results = self.retriever.batch_search(input_query)
        dataset.update_output('retrieval_result', retrieval_results)
        # print('retrieval_results: ',retrieval_results)

        if self.refiner:
            input_prompt_flag = self.refiner.input_prompt_flag
            if 'llmlingua' in self.refiner.name and input_prompt_flag:
                # input prompt
                input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, dataset.retrieval_result)
                ]
                dataset.update_output('prompt', input_prompts)
                input_prompts = self.refiner.batch_run(dataset)
            else:
                # input retrieval docs
                refine_results = self.refiner.batch_run(dataset)
                dataset.update_output('refine_result', refine_results)
                input_prompts = [
                    self.prompt_template.get_string(question=q, formatted_reference=r)
                    for q, r in zip(dataset.question, refine_results)
                ]

        else:
            input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, dataset.retrieval_result)
            ]
        dataset.update_output('prompt', input_prompts)
        # print('input_prompts: ',input_prompts)

        if self.use_fid:
            print('Use FiD generation')
            input_prompts = []
            for item in dataset:
                q = item.question
                docs = item.retrieval_result
                input_prompts.append(
                    [q + " " + doc for doc in docs]
                )
        pred_answer_list = self.generator.generate(input_prompts)
        # print(pred_answer_list)
        dataset.update_output("pred",pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset

    # def internet_search(self, query, num_results=3):
    #     # Example using Bing Search API
        
    #     search_url = "https://api.bing.microsoft.com/v7.0/search"
    #     headers = {"Ocp-Apim-Subscription-Key": subscription_key}

    #     for query in queries:
    #         params = {
    #             "q": query,
    #             "count": num_results,  # Setting the number of results per query
    #             "textDecorations": True,
    #             "textFormat": "HTML"
    #         }
    #     params = {"q": query, "textDecorations": True, "textFormat": "HTML",'count': 3}
    #     response = requests.get(search_url, headers=headers, params=params)
    #     response.raise_for_status()
    #     search_results = response.json()
    #     return search_results
    
    def internet_search(self, queries, num_results=3):
        subscription_key = api_azure_key
        search_url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": subscription_key}
        all_search_results = []

        for query in queries:
            params = {
                "q": query,
                "count": num_results,  # Setting the number of results per query
                "textDecorations": True,
                "textFormat": "HTML"
            }
            response = requests.get(search_url, headers=headers, params=params)
            response.raise_for_status()
            search_results = response.json()
            all_search_results.append(search_results)

        return all_search_results
    
    def run_internet_retrieval(self, dataset, do_eval=True, pred_process_fun=None):
        input_queries = dataset.question
        
        # Use the internet search method
        retrieval_results = self.internet_search(input_queries)
        
        all_formatted_results = []
        for query_results in retrieval_results:
            formatted_results = []
            if "webPages" in query_results:
                for result in query_results["webPages"]["value"]:
                    formatted_results.append({
                        'id': result['id'],
                        'contents': result['snippet'],
                        'score': result.get('rank', 'N/A')
                    })
            all_formatted_results.append(formatted_results)
        
        print(all_formatted_results)

        dataset.update_output('retrieval_results', all_formatted_results)

        input_prompts = [
                    self.prompt_template.get_string(question=q, retrieval_result=r)
                    for q, r in zip(dataset.question, all_formatted_results)
            ]
        dataset.update_output('prompt', input_prompts)
        # pred_answer_list = [self.generator.generate(query) for query in input_queries]
        pred_answer_list = self.generator.generate(input_prompts)
        # dataset.update_output("pred",pred_answer_list)
        print(pred_answer_list)
        dataset.update_output("pred", pred_answer_list)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset


    
    # def run_internet_retrieval(self, dataset, do_eval=True, pred_process_fun=None):
    #     input_query = dataset.question
        
    #     # Use the internet search method
    #     retrieval_results = self.internet_search(input_query)
        
    #     # Process retrieval results
    #     formatted_results = []
    #     for result in retrieval_results["webPages"]["value"]:
    #         # print(result.keys())
    #         formatted_results.append({
    #             'id': result['id'],
    #             'contents': result['snippet'],
    #             'score': result.get('rank', 'N/A')
    #         })
        
    #     print(formatted_results)
    #     pred_answer_list = self.generator.generate(input_query)
    #     dataset.update_output("pred", pred_answer_list)

    #     dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

    #     return dataset

class ConditionalPipeline(BasicPipeline):
    def __init__(self, config, prompt_template = None):
        """
        inference stage:
            query -> judger -> sequential pipeline or naive generate
        """

        super().__init__(config, prompt_template)
        self.judger = get_judger(config)

        self.sequential_pipeline = SequentialPipeline(config, prompt_template)
        from flashrag.prompt import PromptTemplate
        self.zero_shot_templete = PromptTemplate(
            config = config,
            system_prompt =  "Answer the question based on your own knowledge. \
                            Only give me the answer and do not output any other words.",
            user_prompt = "Question: {question}"
        )

    def run(self, dataset, do_eval=True, pred_process_fun=None):
        # judge_result: list of bool element, representing whether to use retrieval
        judge_result = self.judger.judge(dataset)
        dataset.update_output('judge_result', judge_result)

        # split dataset based on judge_result
        pos_dataset, neg_dataset = split_dataset(dataset, judge_result)

        pos_dataset = self.sequential_pipeline.run(pos_dataset, do_eval=False)
        self.sequential_pipeline.prompt_template = self.zero_shot_templete
        neg_dataset = self.sequential_pipeline.naive_run(neg_dataset, do_eval=False)

        # merge datasets into original format
        dataset = merge_dataset(pos_dataset, neg_dataset, judge_result)

        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)

        return dataset




