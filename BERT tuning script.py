import pandas as pd
# importing pandas to deal with the dataset
import torch
# importing torch for deep learning functionality
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# from the Huggingface "transformers" library we import:
# "BertTokenizer" - tokenizer of texts for BERT fine-tuning
# "BertForSequenceClassification" - the BERT model for "sequence classification" (classifying sequcnes of texts into
# categories/classes)
# "Trainer" - a high-level API class that facilitates training LLMs
# "TrainingArguments" - a class that provides configurable arguments for training, such:
# batch size - the size of the data batch that will be used in training
# number of epochs - the number of times we will run the model to update its parameters
# output directory - where the resulting parameters will be uploaded
# evaluation strategy - the method that will be used to evaluating the resulting model
from datasets import Dataset
# a class from the datasets library that's made to handle dataset (namely, convert from a Dataframe to a dataset
# processable by BERT)
from ray import tune
# "Ray Tune" is a library for hyperparameter tuning;
from ray.tune.schedulers import PopulationBasedTraining
# a tool for adaptive hyperparameter tuning
df = pd.read_csv('bert_input.csv')
# we store the prepared dataset under the variabe df as a Dataframe
dataset = Dataset.from_pandas(df)
# we convert it to a special object for BERT training using the Dataset.from_pandas() method
train_test_split = dataset.train_test_split(test_size=0.1)
# we split the created dataset object into the training and testing part using the .train_test_split method of the
# parent class, setting the split ration at 0.1, which means that 90% will be used to training and 10% for testing
# we store the result of the splitting the array object named "train_test_split"
train_dataset = train_test_split['train']
# we take the item stored in the "train_test_split" object under they key "train" into a variable named
# "train_dataset"
eval_dataset = train_test_split['test']
# we take the item stored in the "train_test_split" object under the key "test" into a variable named
# "eval_dataset"
model_name = "bert-base-uncased"
# we store the name of the model we want to train in a variable "model_name"
tokenizer = BertTokenizer.from_pretrained(model_name)
# we store the return of "BertTokenizer"'s class "from_pretrained" method passed the
# parameter "model_name" into a variable named "tokenizer"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=6)
# we store the model itself (not the name) in a variable called "model";
# the stored object is returned by the "from_pretrained(model_name, num_labels = 6)" method of the
# "BertForSequenceClassification" class, passed the following parameters:
# "model_name" - the model that we've chosen, which is the most popular (if I'm not mistaken) BERT
# version named "bert-base-uncased"
# "num_labels = 6" is the parameter that indicates the number of labels that our fine-tuning dataset contains
# so, our dataset labeling is not "binary" :)

def preprocess_function(examples):
    # we're initializing a new function named "preprocess_function" that takes only one parameter: "examples"
    # (must be an array in my understanding)
    return tokenizer(examples['text'], truncation=True, padding=True)
    # we return the text with tokenizer applied and parameters like cutting and extending to maintain
    # constant size turned on (set to True)

train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)
# we preprocess the created datasets for batch training

training_args = TrainingArguments(output_dir="./results", evaluation_strategy="epoch", per_device_train_batch_size=8,
    per_device_eval_batch_size=8, num_train_epochs=3, weight_decay=0.01)
# creating an instance of the TrainingArguments class, we set the arguments for training:
# output_dir="./results" - where our results will be stored
# evaluation_strategy="epoch" - sets the moment of evaluation - at the end of each epoch
# "per_device_train_batch_size=8" - splitting the training set into 8 batches per each CPU/GPU
# "per_device_eval_batch_size=8" - same, but for the evaluation dataset
# "num_train_epochs=3" - train for 3 epoch
# "weight_decay=0.01" - setting the regularisation strength to prevent overfitting

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
# creating an instance of a Hugginface class "Trainer" with the following parameters:
# "model=model" - the model that will be trained: BertforSequenceClassification
# "args=training_args" - the previously created training parameters object
# "train_dataset=train_dataset" - we set the dataset for traning (created earlier from the original dataset)
# "eval_dataset=eval_dataset" - we set the dataset for evaluation (created earlier from the orignial dataset)

def tune_train(config):
    # we define a function to be used by ray for hyperparameter tuning
    args = TrainingArguments(output_dir="./results", evaluation_strategy="epoch", per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"], num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"])

    trainer = Trainer(model=model, args=args, train_dataset=train_dataset,  eval_dataset=eval_dataset)

    trainer.train()

search_space = {"per_device_train_batch_size": tune.choice([8, 16, 32]), "per_device_eval_batch_size": tune.choice([8, 16, 32]),
    "num_train_epochs": tune.choice([3, 4, 5]), "weight_decay": tune.loguniform(1e-4, 1e-1)}
# we define the range in which ray must search for optimal hyperparameters

scheduler = PopulationBasedTraining(time_attr="training_iteration", metric="eval_loss", mode="min")

analysis = tune.run(tune.with_parameters(tune_train), resources_per_trial={"cpu": 1, "gpu": 0}, metric="eval_loss",
                    mode="min", config=search_space, num_samples=10, scheduler=scheduler)
# we run ray's tune to find the best hyperparameters for our fine-tuned model

print("Best hyperparameters found were: ", analysis.best_config)
# we display the found hyperparameters, while the actuals parameters are stored in "./results" output directory