BATCH_SIZE = 64

EPOCHS = 2

WARMUP_STEPS = 100

EVAL_STEPS = 1000

LEARNING_RATE = 2e-5

SBERT_INIT = "Paulanerus/AncientGreekVariantSBERT"  # pranaydeeps/Ancient-Greek-BERT

SBERT_SAVE_PATH = "model/"

DATA_PATH = "data"

RAW_VERSES = f"{DATA_PATH}/verses.csv"
RAW_WORDS = f"{DATA_PATH}/words.csv"
RAW_OCCURRENCES = f"{DATA_PATH}/occurrences.csv"
TEMP_VERSES = f"{DATA_PATH}/temp_verses.csv"

PREPRO_PATH = f"{DATA_PATH}/pairs/"

FEATURES = ["input1", "input2"]

FEATURES += ["Gender"]

N = 1

FEATURE_DIM = 16
