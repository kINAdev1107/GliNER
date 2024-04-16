import json
from gliner import GLiNER

import torch
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import os
import wandb

train_path = 'final_train.json'
eval_path = "final_test.json"

with open(train_path, "r") as f:
    data = json.load(f)
with open(eval_path, "r") as f:
    eval_data = json.load(f)

model_name ="urchade/gliner_large-v2" 
model = GLiNER.from_pretrained(model_name)
name = model_name.split("/")[-1]
filename = f"finetuned-{name}"


wandb.init(project='GLiNER-Exp', name='final-training')

for name, param in model.named_parameters():
    if 'span_rep_layer' in name or 'prompt_rep_layer' in name:
        param.requires_grad = True
    elif 'rnn' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False


from types import SimpleNamespace
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the hyperparameters in a config variable
config = SimpleNamespace(
    num_steps=92450, # number of training iteration
    train_batch_size=2, 
    eval_every=100, # evaluation/saving steps
    save_directory=f"{filename}/logs/", # where to save checkpoints
    warmup_ratio=0.1, # warmup steps
    device=device,
    lr_encoder=1e-5, # learning rate for the backbone
    lr_others=5e-5, # learning rate for other parameters
    freeze_token_rep=False, # freeze of not the backbone
    
    # Parameters for set_sampling_params
    max_types=25, # maximum number of entity types during training
    shuffle_types=True, # if shuffle or not entity types
    random_drop=True, # randomly drop entity types
    max_neg_type_ratio=1, # ratio of positive/negative types, 1 mean 50%/50%, 2 mean 33%/66%, 3 mean 25%/75% ...
    max_len=384 # maximum sentence length
)


def train(model, config, train_data, eval_data=None):
    model = model.to(config.device)

    # Set sampling parameters from config
    model.set_sampling_params(
        max_types=config.max_types, 
        shuffle_types=config.shuffle_types, 
        random_drop=config.random_drop, 
        max_neg_type_ratio=config.max_neg_type_ratio, 
        max_len=config.max_len
    )
    
    model.train()

    # Initialize data loaders
    train_loader = model.create_dataloader(train_data, batch_size=config.train_batch_size, shuffle=True)
    eval_loader = model.create_dataloader(eval_data, batch_size=config.train_batch_size, shuffle=True)

    # Optimizer
    optimizer = model.get_optimizer(config.lr_encoder, config.lr_others, config.freeze_token_rep)

    pbar = tqdm(range(config.num_steps))

    if config.warmup_ratio < 1:
        num_warmup_steps = int(config.num_steps * config.warmup_ratio)
    else:
        num_warmup_steps = int(config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=config.num_steps
    )

    iter_train_loader = iter(train_loader)
    iter_eval_loader = iter(eval_loader)

    temp = 0
    for step in pbar:
            try:
                x = next(iter_train_loader)
                x_eval = next(iter_eval_loader)
            except StopIteration:
                iter_train_loader = iter(train_loader)
                x = next(iter_train_loader)

            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(config.device)
            for k,v in x_eval.items():
                if isinstance(v, torch.Tensor):
                    x_eval[k] = v.to(config.device)

            loss = model(x)  # Forward pass
            eval_loss = model(x_eval)
            
            # Check if loss is nan
            if torch.isnan(loss):
                continue

            loss.backward()  # Compute gradients
            optimizer.step()  # Update parameters
            scheduler.step()  # Update learning rate schedule
            optimizer.zero_grad()  # Reset gradients
            epoch = step // len(train_loader)
            description = f"step: {step} | epoch: {epoch} | loss: {loss.item():.2f} | eval_loss: {eval_loss.item():.2f}"
            pbar.set_description(description)
            wandb.log({"loss": loss.item(), "eval_loss": eval_loss.item()})


            if epoch > temp:
                model.eval()
                
                if eval_data is not None:
                    results, result_dict = model.evaluate(eval_data[:100], flat_ner=True, threshold=0.5, batch_size=5,
                                         entity_types=['B-EMAIL','B-ID_NUM','B-NAME_STUDENT','B-PHONE_NUM',
                                        'B-STREET_ADDRESS','B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM',
                                        'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS','I-URL_PERSONAL']) #,'O'])

                    print(f"Step={step}\n{results}")
                    # wandb.log({"f1": f1})
                    wandb.log(result_dict)
                temp = epoch
                if not os.path.exists(config.save_directory):
                    os.makedirs(config.save_directory)
                torch.cuda.empty_cache()
                model.save_pretrained(f"{config.save_directory}/finetuned_{epoch}")

                model.train()


train(model, config, data, eval_data)

model.save_pretrained(filename)