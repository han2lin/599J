from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, GPT2Config, GPT2ForQuestionAnswering
from transformers import TrainingArguments, Trainer
from torch.utils.data import Dataset
import argparse
import copy
from enum import Enum
import logging
import pandas as pd
import re
import transformers
import torch
import wandb

class ModelSize(Enum):
    MEDIUM = 1
    LARGE = 2


def encode(examples, tokenizer):
    contexes = examples["context"]
    questions = examples["question"]
    answers = examples["answers"]
    samples = [f"""Context: {context}
Question: {question}
Answer: {answer['text'][0]}{tokenizer.eos_token}""" for context, question, answer in zip(contexes, questions, answers)]
    return tokenizer(samples, truncation=True, padding="max_length")



def get_datasets(tokenizer, dataset="han2lin/squad", cache_dir=None):
    all_datasets = load_dataset(dataset, cache_dir=cache_dir)
    train_dataset = all_datasets['train']
    valid_dataset = all_datasets['valid']

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_cache_file_name, valid_cache_file_name = None, None
    if cache_dir:
        train_cache_file_name = f"{cache_dir}/{tokenizer.name_or_path}_train_encoded"
        valid_cache_file_name = f"{cache_dir}/{tokenizer.name_or_path}_valid_encoded"
    logging.info(f"Dataset cache files: {train_cache_file_name} and {valid_cache_file_name}")
    train_dataset = train_dataset.map(lambda x: encode(x, tokenizer), 
                                      batched=True,
                                      cache_file_name=train_cache_file_name)
    valid_dataset = valid_dataset.map(lambda x: encode(x, tokenizer), 
                                      batched=True,
                                      cache_file_name=valid_cache_file_name)

    return train_dataset, valid_dataset


def fine_tune_gpt2(model, 
                   tokenizer, 
                   train_dataset, 
                   valid_dataset,
                   logging_steps,
                   save_steps,
                   per_device_train_batch_size,
                   per_device_eval_batch_size,
                   epochs,
                   learning_rate,
                   train_output_dir,
                   save_model_dir,
                   report_to="all"):

    logging.info("Fine tuning parameters:")
    for label, param in zip(["logging_steps", "save_steps", "batch_size", "epochs", "learning_rate"],
                            [logging_steps, save_steps, batch_size, epochs, learning_rate]):
        logging.info(f"\t{label}: {param}")
        
    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)
    
    # Set training arguments
    training_args = TrainingArguments(
        output_dir = train_output_dir, 
        evaluation_strategy = "steps", 
        disable_tqdm = False,
        logging_steps = logging_steps,
        logging_strategy = "steps",
        save_steps = save_steps,
        num_train_epochs = epochs,
        per_device_train_batch_size = per_device_train_batch_size,
        per_device_eval_batch_size = per_device_eval_batch_size,
        learning_rate = learning_rate,
        # optim="paged_adamw_32bit",
        report_to = report_to,
    )
    
    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    trainer.train()
    log_history = pd.DataFrame(trainer.state.log_history)
    logging.info(log_history)
    
    # Save the fine-tuned model
    model.save_pretrained(save_model_dir)
    tokenizer.save_pretrained(save_model_dir)


def model_size_string(value):
    if value.lower() in ["l", "large"]:
        return ModelSize.LARGE
    elif value.lower() in ["m", "med", "medium"]:
        return ModelSize.MEDIUM
    else:
        raise argparse.ArgumentTypeError(f"Size should be either 'medium' or 'large' but received {value}")


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
    parser = argparse.ArgumentParser(description="Fine-tune GPT 2")
    
    parser.add_argument(
        "--use_wandb",
        dest="use_wandb",
        type=bool,
        default=False,
        help="Set to True to save training run in wandb",
    )
    parser.add_argument(
        "--model_size",
        dest="model_size",
        type=model_size_string,
        default="large",
        help="GPT2 size to fine-tune ('medium' or 'large'). Defaults to large.",
    )
    parser.add_argument(
        "--model_path",
        dest="model_path",
        type=str,
        default="gpt2-large",
        help="GPT2 model path for retrieving weights to fine-tune",
    )
    parser.add_argument(
        "--cache_dir",
        dest="cache_dir",
        type=str,
        required=False,
        help="Directory for caching model weights, tokens, etc. Useful if running on hyak.",
    )


    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int_range(1,64000),
        default=2,
        help="Batch size for fine tuning (this is the *per_device* batch size)",
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        type=int_range(1,15),
        default=1,
        help="Epochs of fine tuning",
    )
    parser.add_argument(
        "--lr",
        dest="learning_rate",
        type=float,
        default=1e-5,
        help="Epochs of fine tuning",
    )
    parser.add_argument(
        "--logging_steps",
        dest="logging_steps",
        type=int,
        default=1000,
        help="Logging steps for fine tuning",
    )
    parser.add_argument(
        "--save_steps",
        dest="save_steps",
        type=int,
        default=1000,
        help="Save steps for fine tuning",
    )
    known_args, pipeline_args = parser.parse_known_args(argv)

    # Model args
    model_size = known_args.model_size
    model_path = known_args.model_path

    logging.info(f"Fine tuning {model_path} of size {model_size}")

    name = model_path.split('/')[-1]
    output_dir = f"ft_log_{name}"
    save_dir = f"ft_model_{name}"
    cache_dir = known_args.cache_dir
    
    if cache_dir: 
        logging.info(f"Using cache_dir={cache_dir}")

    #  Fine tuning args
    batch_size = known_args.batch_size
    epochs = known_args.epochs
    learning_rate = known_args.learning_rate
    logging_steps = known_args.logging_steps
    save_steps = known_args.save_steps
    report_to = "all"

    if known_args.use_wandb:
        wandb.login()

        wandb.init(
            project=name,
            config={
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "dataset": "SQuAD",
            },
        )
        report_to = "wandb"
    
    transformers.logging.set_verbosity_error()

    tokenizer = None
    if model_size == ModelSize.MEDIUM:
        tokenizer = AutoTokenizer.from_pretrained("gpt2-medium", cache_dir=cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2-large", cache_dir=cache_dir)

    train_dataset, valid_dataset = get_datasets(tokenizer, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)

    fine_tune_gpt2(model, 
                   tokenizer, 
                   train_dataset['input_ids'], 
                   valid_dataset['input_ids'],
                   logging_steps,
                   save_steps,
                   batch_size,
                   batch_size,
                   epochs,
                   learning_rate,
                   output_dir,
                   save_dir,
                   report_to)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.getLogger().setLevel(logging.INFO)
    main()