from sentence_transformers import SentenceTransformer

import config
import utils.prediction_helpers as ph
from utils.check_cuda import check_cuda_and_gpus
from utils.string_norm import strip_accents_and_lowercase

device = check_cuda_and_gpus()
print(f"Device: {device}")

model = SentenceTransformer("./" + config.SBERT_SAVE_PATH + "/", device=device)

# Sentence, Gender Score

xsent = [
    "ουαι υμιν γραμματεις και φαρισαιοι υποκριται οτι καθαριζετε το εξωθεν του ποτηριου και της παροψιδος εσωθεν δε γεμουσιν αρπαγης και αδικιας",  # 0.5
    "στελλομενοι τουτο μη τις ημας μωμησηται εν τη αδροτητι ταυτη τη διακονουμενη υφ ημων",  # 0.0
    "γεγραπται γαρ οτι αβρααμ δυο υιους εσχεν ενα εκ της παιδισκης και ενα εκ της ελευθερας",  # 1.0
    "η δε απεκριθη και λεγει αυτω ναι κε και γαρ τα κυναρια υποκατω της τραπεζης εσθιει απο των ψιχιων των παιδιων",  # 0.0
]

ysent = [
    "εν ω εχομεν την απολυτρωσιν δια του αιματος αυτου την αφεσιν των αμαρτιων",  # 0.5
    "μολις γαρ υπερ δικαιου τις αποθανειται υπερ γαρ του αγαθου ταχα τις και τολμα αποθανειν",  # 0.0
    "παραγενομενοι δε προς αυτον οι ανδρες ειπον ιωαννης ο βαπτιστης απεσταλκεν ημας προς σε λεγων συ ει ο ερχομενος η αλλον προσδοκωμεν",  # 1.0,
    "η δε απεκριθη και λεγει αυτω ναι κε και γαρ τα κυναρια υποκατω της τραπεζης εσθιουσιν απο των ψιχιων των παιδιων",  # 0.0
]

xsent_norm = [strip_accents_and_lowercase(s) for s in xsent]
ysent_norm = [strip_accents_and_lowercase(s) for s in ysent]

xsent_encoded = model.encode(xsent_norm)
ysent_encoded = model.encode(ysent_norm)

preds = ph.get_preds(
    xsent_encoded,
    ysent_encoded,
    biases=None,
    n=config.N,
    dim=config.FEATURE_DIM,
)

for i, x in enumerate(xsent):
    sims = preds[i]
    print(f"Global: {sims[0]}, Gender: {sims[1]}\n")
