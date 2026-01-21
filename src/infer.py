from sentence_transformers import SentenceTransformer

import config
import utils.prediction_helpers as ph
from utils.check_cuda import check_cuda_and_gpus
from utils.string_norm import strip_accents_and_lowercase

device = check_cuda_and_gpus()
print(f"Device: {device}")

model = SentenceTransformer("./" + config.SBERT_SAVE_PATH + "/", device=device)

# Sentence, Gender

xsent = [
    "διανοιγων και παρατιθεμενος οτι τον χν εδει παθειν και αναστηναι εκ νεκρων και οτι ουτος εστιν ις χς ον εγω καταγγελλω υμιν",  # m
    "τουτο γαρ εστιν το θελημα του πεμψαντος με πρς ινα πας ο θεωρων τον υιον και πιστευων εις αυτον εχει ζωην αιωνιον και εγω αναστησω αυτον εν τη εσχατη ημερα",  # u
    "και καθως προειρηκεν ησαιας ει μη κς σαβαωθ εγκατελειπε σπερμα ως σοδομα αν εγενηθημεν και ως γομορρα αν ωμοιωθημεν",  # m
    "τοτε λεγει αυτω ο ις υπαγε οπισω μου σατανα κν τον θν σου προσκυνησεις και αυτω μονω λατρευσεις",  # m
]

ysent = [
    "ευρε δε εκει ανον τινα ονοματι αινεα εξ ετων οκτω κατακειμενον επι κραβαττω ος ην παραλελυμενος",  # m
    "και εν τη οικεια παλιν οι μαθηται αυτου περι του αυτου επηρωτησαν αυτον",  # u
    "κατα συγκυριαν δε ιερευς τις κατεβαινεν τη οδωι εκεινηι και ιδων αντιπαρηλθεν",  # u
    "τω καιρω εκεινω μαρια ιστικη προς το μνημειον κλαιουσα εξω ως ουν εκλαιεν παρεκυψεν εις το μνιμειον",  # f
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
    print(f"Global: {sims[0]:.4f}, Gender: {sims[1]:.4f}\n")
