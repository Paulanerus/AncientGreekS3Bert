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
    "γνωριζω υμιν το ευαγγελιον το ευηγγελεισθεν υπ εμου οτι ουκ εστι κατα ανον",  # 0.0
    "και απηλθεν παλιν περα  του ιορδανου εις τον τοπον οπου ην ιωαννης το πρωτον βαπτιζων και εμεινεν εκει",  # 0.5
    "και επατηθη ο λινος εξω της πολεως και εξηλθεν αιμα εκ της λινου αχρι των χαλινων των ιππων απο σταδιων χιλιων εξακοσιων",  # 0.5
    "απειγγειλεν τε ημιν και αυτος πως ιδεν τον αγγελον εν τω οικω αυτου σταθεντα και ειποντα αυτου αποστειλον εις ιωππη ανδρας και μεταπεμψαι σιμωνα τον επικαλουμενον πετρον",  # 1.0
]

ysent = [
    "τι γαρ ει ηπιστησαν τινες μη η απιστια αυτων την πιστιν του θυ κατηγορησε",  # 0.0
    "και ιδου τινες των γραμματαιων ειπαν εν εαυτοις ουτος βλασφημει",  # 0.5
    "οι δε λοιποι κρατησαντες τους δουλους αυτου εδειραν και απεκτειναν",  # 0.5
    "ουχι ταυτα εδει παθειν τον χριστον και εισελθειν εις την δοξαν αυτου",  # 1.0
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
