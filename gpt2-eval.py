from datasets import load_dataset
from enum import Enum
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.data.metrics import squad_metrics
from transformers.data.processors import squad
from torch.utils.data import Dataset
import argparse
import logging
import math
import pandas as pd
import tqdm
import transformers
import torch


class ModelSize(Enum):
    MEDIUM = 1
    LARGE = 2

def make_prompts(examples, context_label="context"):
    """Make QA prompts from examples"""
    contexts = examples[context_label]
    questions = examples["question"]
    samples = [f"""Context: {context}
Question: {question}
Answer:""" for context, question in zip(contexts, questions)]
    return {"prompt": samples}


def get_test_dataset(tokenizer, 
                     dataset="han2lin/squad", 
                     cache_dir=None,
                     context_label="context"):
    logging.info(f"Loading test dataset with cache_dir: {cache_dir}")
    all_datasets = load_dataset(dataset, cache_dir=cache_dir)
    test_dataset = all_datasets["test"]
    logging.info(f"Test dataset size: {len(test_dataset)}")
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    test_cache_file_name = None
    if cache_dir:
        test_cache_file_name = f"{cache_dir}/{tokenizer.name_or_path}_test_encoded"
    logging.info(f"Dataset cache file: {test_cache_file_name}")
    test_dataset = test_dataset.map(
        lambda x: make_prompts(x, context_label=context_label), 
        batched=True,
        cache_file_name=test_cache_file_name,
    )
    
    return test_dataset


def split_batches(data, n):
    for i in range(0, len(data), n):
        yield data[i : i + n]

def get_predictions(prompts, 
                    model, 
                    tokenizer, 
                    max_new_tokens=50,
                    use_cuda=True):
    input_ids = tokenizer(prompts,
                          max_length=1024,
                          padding="max_length", 
                          return_tensors="pt")
    if use_cuda:
        input_ids.to("cuda")
    greedy_output = model.generate(**input_ids, max_new_tokens=max_new_tokens)
    predictions = tokenizer.batch_decode(greedy_output, skip_special_tokens=True)
    del input_ids, greedy_output
    torch.cuda.empty_cache()
    return predictions


def extract_answer(s, tokenizer=None):
    key = "Answer: "
    answer = s[s.find(key):]
    answer_end = len(answer)
    
    ends = [".", "\n"]
    if tokenizer and tokenizer.eos_token:
        ends.append(tokenizer.eos_token)
    
    for e in ends:
        idx = answer.find(e)
        if idx > 0 and idx < answer_end:
            answer_end = idx
    
    return answer[len(key):answer_end]


def to_squad_examples(dataset_pd):
    examples = []
    for _, sample in dataset_pd.iterrows():
        answers = [{"text": t, 
                    "answer_start": s} for t, s in zip(sample["answers"]["text"], 
                                                       sample["answers"]["answer_start"])]
        examples.append(squad.SquadExample(
            qas_id=sample["id"],
            question_text=sample["question"],
            context_text=sample["context"],
            answer_text=None,
            start_position_character=None,
            title=sample["title"],
            answers=answers,
        ))
    return examples    

def score_data(dataset_pd):
    predictions_dict = {}
    for _, (id, prediction) in dataset_pd[["id", "prediction"]].iterrows():
        predictions_dict[id] = prediction
    examples = to_squad_examples(dataset_pd)
    return squad_metrics.get_raw_scores(examples, predictions_dict)


def write_stats(dataset_pd: pd.DataFrame, 
                model_name: str, 
                output_dir: str,
                suffix: str = ""):
    stats_path = f"{output_dir}/{model_name}_stats.csv"
    predictions_path = f"{output_dir}/{model_name}_pred.csv"
    if suffix:
        stats_path = f"{output_dir}/{model_name}_{suffix}_stats.csv"
        predictions_path = f"{output_dir}/{model_name}_{suffix}_pred.csv"

    stats = dataset_pd[["exact_score", "f1_score"]].mean()
    logging.info(f"Overall stats:\n{stats}")
    stats.to_csv(stats_path, header=False)
    dataset_pd.to_csv(predictions_path, columns=["id", "prediction", "exact_score", "f1_score"])


def model_size_string(value):
    if value.lower() in ["l", "large"]:
        return ModelSize.LARGE
    elif value.lower() in ["m", "med", "medium"]:
        return ModelSize.MEDIUM
    else:
        raise argparse.ArgumentTypeError(f"Size should be either 'medium' or 'large' but received {value}")

def perturbation_string(value):
    options = ["", "context", "butter", "yoda", "sentence_reorder"]
    if value is None:
        return ""
    if value.lower() in options:
        return value.lower()
    else:
        raise argparse.ArgumentTypeError(f"Size should be one of {options} but received {value}")
        

def int_range(min, max):
    def int_range_checker(arg):
        try:
            i = int(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("int expected")
        if i < min or i > max:
            raise argparse.ArgumentTypeError(
                "int needs to be in [" + str(min) + ", " + str(max) + "]"
            )
        return i
    return int_range_checker


def main(argv=None):
    parser = argparse.ArgumentParser(description="Zero-shot SQuAD evals for GPT 2")
    
    parser.add_argument(
        "--cuda",
        dest="use_cuda",
        default=True,
        action="store_true",
        help="Use Cuda for eval",
    )
    parser.add_argument(
        "--no-cuda",
        dest="use_cuda",
        action="store_false",
        help="Do not use Cuda",
    )
    parser.add_argument(
        "--model_size",
        dest="model_size",
        type=model_size_string,
        default="large",
        help="GPT2 size to evaluate ('medium' or 'large'). Defaults to large.",
    )
    parser.add_argument(
        "--model_path",
        dest="model_path",
        type=str,
        action="append",
        required=True,
        help="GPT2 models to evaluate",
    )
    parser.add_argument(
        "--dataset_path",
        dest="dataset_path",
        type=str,
        default="han2lin/squad",
        help="Dataset to grab test split from for eval. Defaults to han2lin/squad.",
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int_range(1,64000),
        default=250,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--cache_dir",
        dest="cache_dir",
        type=str,
        required=False,
        help="Directory for caching model weights, tokens, etc. Useful if running on hyak.",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        required=False,
        help="Directory for storing predictions. If not set, uses cache_dir if cache_dir is set, else writes to current directory.",
    )
    parser.add_argument(
        "--perturbation",
        dest="perturbation",
        type=str,
        required=False,
        default="",
        help="The perturbation type to test. If empty, no perturbation is choosen from the test dataset.",
    )


    known_args, pipeline_args = parser.parse_known_args(argv)

    # Model args
    model_size = known_args.model_size
    model_paths = known_args.model_path

    # Dataset
    dataset_path = known_args.dataset_path
    context_label = "context"
    if known_args.perturbation:
        context_label = known_args.perturbation

    logging.info(f"Evaluating models of size {model_size} on {dataset_path}: {model_paths}")
    if context_label != "context":
        logging.info(f"Evaluating with perturbation '{context_label}' from {dataset_path}")

    use_cuda = known_args.use_cuda
    if use_cuda:
        logging.info(f"Using cuda")

    cache_dir = known_args.cache_dir
    output_dir = known_args.output_dir
    if cache_dir: 
        logging.info(f"Using cache_dir={cache_dir}")
    if not output_dir:
        output_dir = cache_dir
    
    tokenizer = None
    if model_size == ModelSize.MEDIUM:
        tokenizer = AutoTokenizer.from_pretrained("gpt2-medium", cache_dir=cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2-large", cache_dir=cache_dir)

    test_dataset = get_test_dataset(tokenizer, 
                                    dataset_path, 
                                    cache_dir=cache_dir,
                                    context_label=context_label)

    for model_path in tqdm.tqdm(model_paths):
        logging.info(f"Evaluating model: {model_path}")
        name = model_path.split("/")[-1]
        model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                     cache_dir=cache_dir)
        if use_cuda:
            model.to("cuda")
        
        dataset_pd = pd.DataFrame(test_dataset)
    
        batch_size = known_args.batch_size
        total_batches = math.ceil(len(dataset_pd["prompt"]) / batch_size)
        full_predictions = []
        for batch in tqdm.tqdm(split_batches(dataset_pd["prompt"].tolist(), batch_size), 
                               total=total_batches):
            full_predictions += get_predictions(batch, model, tokenizer, use_cuda=use_cuda)
        
        dataset_pd["prediction"] = [extract_answer(p) for p in full_predictions]
        exact_scores, f1_scores = score_data(dataset_pd)
        dataset_pd["exact_score"] = exact_scores.values()
        dataset_pd["f1_score"] = f1_scores.values()
    
        logging.info(f"Writing prediction and stats into {output_dir}")
        write_stats(dataset_pd, name, output_dir, suffix=context_label)
        
        del dataset_pd


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.getLogger().setLevel(logging.INFO)
    main()