import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer

import config
import utils.custom_evaluators as evaluation
import utils.custom_losses as losses
import utils.model_freeze as freeze
from utils.check_cuda import check_cuda_and_gpus

device = check_cuda_and_gpus()
print(f"Device: {device}")

model = SentenceTransformer(config.SBERT_INIT, device=device)

teacher = SentenceTransformer(config.SBERT_INIT, device=device)

freeze.freeze_except_last_layers(model, 2)
freeze.freeze_all_layers(teacher)

train_df = pd.read_csv(f"{config.PREPRO_PATH}/train_pairs.csv")
dev_df = pd.read_csv(f"{config.PREPRO_PATH}/dev_pairs.csv")

train_examples = []
for _, row in train_df.iterrows():
    ex = InputExample(
        texts=[row["sentence1"], row["sentence2"]],
        label=[row["gender_score"]],
    )
    train_examples.append(ex)

dev_examples = []
for _, row in dev_df.iterrows():
    ex = InputExample(
        texts=[row["sentence1"], row["sentence2"]],
        label=[row["gender_score"]],
    )
    dev_examples.append(ex)

train_dataloader = torch.utils.data.DataLoader(
    train_examples, shuffle=True, batch_size=config.BATCH_SIZE
)
dev_dataloader = torch.utils.data.DataLoader(
    dev_examples, shuffle=False, batch_size=config.BATCH_SIZE
)

# init losses
distill_loss = losses.DistilLoss(
    model,
    sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
    num_labels=config.N,
    feature_dim=config.FEATURE_DIM,
    bias_inits=None,
)

teacher_loss = losses.MultipleConsistencyLoss(model, teacher)

evaluator = evaluation.DistilConsistencyEvaluator(
    dev_dataloader, loss_model_distil=distill_loss, loss_model_consistency=teacher_loss
)

model.fit(
    train_objectives=[
        (train_dataloader, teacher_loss),
        (train_dataloader, distill_loss),
    ],
    optimizer_params={"lr": config.LEARNING_RATE},
    epochs=config.EPOCHS,
    warmup_steps=config.WARMUP_STEPS,
    evaluator=evaluator,
    evaluation_steps=config.EVAL_STEPS,
    output_path=config.SBERT_SAVE_PATH,
    save_best_model=True,
)
