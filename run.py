import re
import argparse
from similarity_metrics import return_top_k
from model import render_prompt
from model import Sabia_Generator

def get_args():
    parser = argparse.ArgumentParser(
        description="Example script using argparse."
    )

    # ---- Required argument ----
    parser.add_argument(
        "--reference-path",
        type=str,
        required=True,
        help="Reference path"
    )

    # ---- Optional argument with default ----
    parser.add_argument(
        "--train-path",
        type=str,
        default="",
        help="Training path"
    )

    # ---- Optional boolean flag ----
    parser.add_argument(
        "--n-shots",
        type=int,
        default=0,
        help="Number of examples we will use"
    )
    
    # ---- Similarity ----
    parser.add_argument(
        "--similarity",
        type=str,
        choices=["random", "lemma", "stem", "cosine", "smatch-recall", "smatch-f1"],
        default="lemma",
        help="Similarity for selecting n-shots"
    )    

    # ---- Choice of values ----
    parser.add_argument(
        "--model",
        type=str,
        choices=["sabia-7b", "cabrita"],
        default="sabia-7b",
        help="Execution mode."
    )
    

    # ---- Prompt path ----
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Prompt path"
    )    

    return parser.parse_args()

def read_amr_file(path):

    def remove_amr_alignments(amr_str: str) -> str:
        """
        Removes JAMR/ISI alignment tags like '~e.4', '~.e4', '~e4', '~a1', etc.
        """
        return re.sub(r'~[A-Za-z0-9\.\-]+', '', amr_str)    
    
    def extract_amr_snt(instance):
        snt = instance.split("\n")[1].replace("# ::tok-pt ", "")
        amr = " ".join((instance.split("\n")[5:]))
        return {"sent":snt,
                "amr": remove_amr_alignments(amr)
            }
    
    data = None
    with open(path, "r", encoding="utf-8") as f:
        data = f.read().split("\n\n")
        data = [extract_amr_snt(e) for e in data if e.strip()!= ""]
    
    return data



def main():
    args = get_args()
    
    reference_data = read_amr_file(args.reference_path)
    train_data = read_amr_file(args.train_path)
    
    field = "sent"
    if args.similarity in ["smatch-recall", "smatch-f1"]:
        field = "amr"
    
    model = None
    if args.model == "sabia-7b":
        model = Sabia_Generator()
    
    for reference in reference_data:
        sample = ""
        top_results = []
        if args.n_shots > 0:
            top_results = return_top_k(reference, train_data, field, args.n_shots, args.similarity)

        if len(top_results) > 0:
            sample = "\n\n".join(["AMR: " + res[0]["amr"] + "\nResposta: " + res[0]["sent"] for res in top_results])
            sample += "\n\n"
        sample += "AMR: " + reference["amr"] + "\nResposta: "
        
        print(sample.strip())
        
        instruction = render_prompt(args.prompt, sample)
        response = model.evaluate(instruction)
        print(response)
        
        break


if __name__ == "__main__":
    main()
