import os
import sys
import nnsight
nnsight.CONFIG.API.APIKEY =  os.environ["NDIF_TOKEN"]
from nnsight import LanguageModel

from lib.models import create_model
from lib.igrad import plot_igrads, get_igrads_multiple


manual_prompts = [
    ("math", ("5 + 7 + 5 + 3 + 1 + 7 = ", "28")),
    ("question2", ("The spouse of the performer of Imagine is", " Yoko Ono"))
]

def main():
    N_EXAMPLES = 10

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        raise ValueError("Please provide a model name")
    
    llm = create_model(model_name)
    target_dir = "out/igrad"

    llm.eval()

    os.makedirs(target_dir, exist_ok=True)

    # igrads, tokens = get_igrads(llm, "5 + 7 + 5 + 3 + 1 + 7 = ", "28", what="layer")

    for what in ["layer", "attention", "mlp"]:
        inputs = [p for _, p in manual_prompts]
        igrads, tokens = get_igrads_multiple(llm, inputs, what)

        for i, (igrad, token) in enumerate(zip(igrads, tokens)):
            pname = manual_prompts[i][0]
            fig = plot_igrads(igrad, token)
            fig.savefig(os.path.join(target_dir, f"{model_name}_igrad_{what}_{pname}.pdf"), bbox_inches="tight")

    
if __name__ == "__main__":
    main()