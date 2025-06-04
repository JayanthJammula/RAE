import torch
import matplotlib.pyplot as plt
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np
import logging
import argparse
import time

from utils_func import *
from wiki_api.strings import question_token
from model import Extract, Prune

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",           type=str,   required=True)
    parser.add_argument("--dataset",         type=str,   required=True)
    parser.add_argument("--relation_path",   type=str,   required=True)
    parser.add_argument("--NatureL",         action="store_true")
    parser.add_argument("--template",        action="store_true")
    parser.add_argument("--template_number", type=int,   default=5)
    parser.add_argument("--entropy_template_number", type=int, default=5)
    parser.add_argument("--starting_line",   type=int,   default=0)
    parser.add_argument("--mode",            type=str,   choices=["beam","sample"], default="beam")
    parser.add_argument("--correctConflict", action="store_true")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--device",          type=str,   default="cpu")
    parser.add_argument("--conf_threshold",  type=float, default=0.7)
    parser.add_argument("--max_retrieval_rounds", type=int, default=3)
    parser.add_argument("--loss",            type=str, choices=["prob_div","prob_div_log"], default="prob_div")
    parser.add_argument("--beam_width",      type=int, default=5)
    parser.add_argument("--num_beams",       type=int, default=1)
    parser.add_argument("--max_new_tokens",  type=int, default=50)
    parser.add_argument("--temp",            type=float, default=1.0)
    return parser.parse_args()

def retrieval_confidence(retrieved: str, ground: set) -> float:
    if not ground: return 0.0
    hits = sum(1 for fact in ground if fact in retrieved)
    return hits / len(ground)

def log_metrics(metrics):
    total = metrics['total_ques']
    logger.info(f"raw_exact_match_acc:  {metrics['raw_exact_match_cor']/total:.4f}")
    logger.info(f"raw_par_match_acc:    {metrics['raw_par_match_cor']/total:.4f}")
    logger.info(f"prun_exact_match_acc: {metrics['prun_exact_match_cor']/total:.4f}")
    logger.info(f"prun_par_match_acc:   {metrics['prun_par_match_cor']/total:.4f}")
    logger.info(f"raw_ans_acc:          {metrics['total_raw_cor']/total:.4f}")
    logger.info(f"prun_ans_acc:         {metrics['total_prun_cor']/total:.4f}")


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)

    MODEL_CONFIGS = {
        "gpt2":   "gpt2-medium",
        "vicuna": "lmsys/vicuna-7b-v1.1",
        "llama2": "meta-llama/Llama-2-7b-chat-hf",
        "falcon": "tiiuae/falcon-7b",
    }
    model = AutoModelForCausalLM.from_pretrained(MODEL_CONFIGS[args.model]).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS[args.model])
    model.eval()
    logger.info("Loaded model")

    lines      = load_dataset(f"data/{args.dataset}.json")
    train      = load_dataset("data/MQuAKE-CF.json")
    tuple_list = load_train_question("data/train_question_tuple.txt")
    edit_list  = build_fact(lines)
    rel_lines  = load_dataset(args.relation_path)
    if args.NatureL:
        NL_dict = load_dataset("data/cloze_templates_NL.json")
        relation_dict, revserse_dict = build_relation(rel_lines, NL_dict)
    else:
        relation_dict, revserse_dict = build_relation(rel_lines, None)
    triplets_dict      = load_triplets_dict("data/Wikidata_triplets_dict_MQuAKE-T.pkl")
    orig_triplets_dict = load_triplets_dict("data/Wikidata_triplets_dict_Edited_T.pkl")

    extractor = Extract(model, tokenizer,
                        triplets_dict, relation_dict,
                        revserse_dict, orig_triplets_dict, args)
    pruner    = Prune(model, tokenizer, args)

    metrics = {k:0 for k in [
        'total_ques','raw_exact_match_cor','raw_par_match_cor',
        'prun_exact_match_cor','prun_par_match_cor',
        'total_raw_cor','total_prun_cor']}
    correct_viz = 0
    wrong_viz   = 0
    K_heads     = 5

    t0 = time.time()

    for i, line in enumerate(lines):
        if i < args.starting_line: continue
        metrics['total_ques'] += 1
        case_metrics = {k:0 for k in metrics if k!='total_ques'}

        for j in range(3):
            # build ground truth
            ground = set()
            for l, tri in enumerate(line["orig"]['new_triples_labeled']):
                if args.NatureL:
                    s = ' '.join([tri[0], NL_dict[line["orig"]["new_triples"][l][1]], tri[2]])
                else:
                    s = ' '.join(tri)
                ground.add(s)
            question = line['questions'][j]

            # build prompts
            if args.template:
                prob_ex = tempalate_extractor(question, args.NatureL, 'prob')
                prom_questions = prob_ex + f"Question: {question}\nAnswer:"
                ans_prompt     = tempalate_extractor(question, args.NatureL, 'ans')
            else:
                prom_questions = question
                ans_prompt     = question

            # adaptive retrieval
            raw_ent = ner_entity(question)
            retrieved = ""
            for rnd in range(1, args.max_retrieval_rounds+1):
                retrieved = extractor.multi_hop_search(
                    prom_questions, raw_ent,
                    len(ground)+2, [], rounds=rnd
                )
                conf = retrieval_confidence(retrieved, ground)
                logger.info(f"Case{i+1} Q{j} round{rnd} conf={conf:.2f}")
                if conf >= args.conf_threshold:
                    break

            pruned = pruner.prune_fact(question, retrieved, ans_prompt)

            # QA pruned
            prun_ans, prun_cor = QA_func(
                model, tokenizer, line,
                pruned, question, ans_prompt,
                'Pruned'
            )
            is_correct = prun_cor > 0

            # collect attentions once if needed
            if (is_correct and correct_viz < 5) or (not is_correct and wrong_viz < 5):
                inputs = tokenizer(prom_questions, return_tensors="pt").to(args.device)
                model.config.output_attentions = True
                gen_out = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    output_attentions=True,
                    return_dict_in_generate=True,
                    do_sample=False,
                    num_beams=args.num_beams,
                    temperature=args.temp
                )
                step_attns = gen_out.attentions[-1] 

                prompt_len = inputs.input_ids.shape[1]
                scores = []
                for l_idx, layer in enumerate(step_attns):
                    for h_idx in range(layer.shape[1]):
                        mat = layer[0, h_idx].cpu()
                        scores.append(((l_idx, h_idx), mat[-1, :prompt_len].sum().item()))
                top_k = sorted(scores, key=lambda x: x[1], reverse=True)[:K_heads]

                # plot only top_k
                for (l_idx, h_idx), _ in top_k:
                    attn_mat = step_attns[l_idx][0, h_idx].cpu().detach()
                    toks = tokenizer.convert_ids_to_tokens(gen_out.sequences[0])
                    plt.figure(figsize=(4,4))
                    plt.imshow(attn_mat, aspect="auto")
                    plt.xticks(range(len(toks)), toks, rotation=90)
                    plt.yticks(range(len(toks)), toks)
                    plt.title(f"Case{i+1} Q{j} L{l_idx} H{h_idx}")
                    plt.tight_layout()
                    fname = f"attn_c{i+1}_q{j}_L{l_idx}_H{h_idx}.png"
                    plt.savefig(fname)
                    plt.close()
                    logger.info(f"Saved {fname}")

                if is_correct: correct_viz += 1
                else:          wrong_viz   += 1

            # QA raw & matching for metrics
            raw_ans, raw_cor = QA_func(
                model, tokenizer, line,
                retrieved, question, ans_prompt,
                'Raw'
            )
            raw_pm, raw_em = match_func(ground, retrieved, 'Raw')
            prun_pm, prun_em = match_func(ground, pruned, 'Pruned')

            case_metrics['raw_par_match_cor']   += raw_pm
            case_metrics['raw_exact_match_cor'] += raw_em
            case_metrics['prun_par_match_cor']  += prun_pm
            case_metrics['prun_exact_match_cor']+= prun_em
            case_metrics['total_raw_cor']       += raw_cor
            case_metrics['total_prun_cor']      += prun_cor

            if case_metrics['total_prun_cor'] > 0:
                break

        for k, v in case_metrics.items():
            if v > 0: metrics[k] += 1

        if (i+1) % 10 == 0:
            logger.info(f"Progress {i+1}")
            log_metrics(metrics)

        if correct_viz>=5 and wrong_viz>=5:
            logger.info("Collected 5 correct & 5 wrong attentions; stopping.")
            break

    t1 = time.time()
    logger.info(f"Total runtime: {(t1-t0):.1f}s")
    logger.info("=== FINAL METRICS ===")
    log_metrics(metrics)
