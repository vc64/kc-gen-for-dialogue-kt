import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dialogue_kt.data_loading import get_kc_result_filename, get_qual_result_filename, get_default_fold

def visualize_single(kc_info):
    target_dialogue_idx = list(kc_info.keys())[3] # 3, tried up to 9
    dialogue = kc_info[target_dialogue_idx]
    kc_to_curve = {}
    for turn_idx, turn in enumerate(dialogue):
        for kc, prob in turn.items():
            kc_to_curve.setdefault(kc, {"x": [], "y": []})
            kc_to_curve[kc]["x"].append(turn_idx)
            kc_to_curve[kc]["y"].append(prob)
    plt.rcParams["figure.figsize"] = (12,8)
    for kc, curve in kc_to_curve.items():
        plt.plot(curve["x"], curve["y"], label=kc)
    plt.legend()
    plt.show()

def visualize_average(kc_info, args):
    plot_deltas = True
    plot_means = True
    kc_to_probs = {}
    for dialogue in kc_info.values():
        # Get sequential list of predicted probabilities for each KC in this dialogue
        dia_kc_to_probs = {}
        for turn in dialogue:
            for kc, prob in turn.items():
                dia_kc_to_probs.setdefault(kc, []).append(prob)
        # Add each of those lists to global running list per KC
        for kc, probs in dia_kc_to_probs.items():
            kc_to_probs.setdefault(kc, []).append(probs)
    # Plot curves for most common KCs
    common_kcs = [kv[0] for kv in sorted(kc_to_probs.items(), key=lambda kv: -len(kv[1]))]
    for kc_idx, kc in enumerate(common_kcs[:15]):
        prob_lists = kc_to_probs[kc]
        if plot_deltas:
            prob_lists = [[prob - probs[0] for prob in probs] for probs in prob_lists]
        gt1_count = sum([len(probs) > 1 for probs in prob_lists])
        lens = [len(probs) for probs in prob_lists]
        print(f"{kc_idx} - {kc}: {len(prob_lists)} dialogues, >1 occurrence: {gt1_count}, avg len: {sum(lens) / len(lens):.2f}")
        plt.rcParams["figure.figsize"] = (9, 6)
        plt.rcParams["font.size"] = 24
        if plot_means:
            means = []
            stds = []
            for i in range(max(lens)):
                cur_idx_probs = np.array([probs[i] for probs in prob_lists if i < len(probs)])
                means.append(cur_idx_probs.mean())
                stds.append(cur_idx_probs.std())
            eb = plt.errorbar(np.arange(len(means)) + 1, means, yerr=stds, marker='o', linewidth=3, capsize=7, ecolor='black')
            eb[-1][0].set_linestyle("--")
        else:
            for probs in prob_lists:
                plt.plot(np.arange(len(probs)), probs)
        plt.grid(True, which="major")
        plt.axis([.5, 15.5, -.62, .62])
        plt.xticks(np.arange(1, 16, 2))
        plt.xlabel("KC Occurrence")
        plt.ylabel("Change in KC Mastery")
        # plt.title("\n".join(wrap(kc.split(";")[0], 70)), fontsize=24)
        # plt.show()
        plt.savefig(f"results/linear_kc_{args.dataset}_{args.tag_src}_{kc_idx}{'_delta' if plot_deltas else ''}.png", dpi=300, bbox_inches="tight")
        plt.close()

def analyze_qual_data(args):
    df = pd.read_csv(get_qual_result_filename(args), dtype=str)
    kc_to_labels = {}
    for _, sample in df.iterrows():
        if isinstance(sample["KCs"], float) or sample["KCs"] == "None":
            continue
        kcs = [kc[3:] for kc in sample["KCs"].split("\n")]
        for kc in kcs:
            kc_to_labels.setdefault(kc, []).append(sample["Correct"] == "true")
    common_kcs = [kv[0] for kv in sorted(kc_to_labels.items(), key=lambda kv: -len(kv[1]))]
    for kc in common_kcs[:20]:
        print(f"{kc}: {np.mean(kc_to_labels[kc]):.2f}")

def visualize(args):
    if args.dataset == "comta":
        kc_info = {}
        for fold in range(1, 6):
            with open(get_kc_result_filename(args, fold)) as kc_file:
                kc_info = {
                    **kc_info,
                    **json.load(kc_file)
                }
    else:
        with open(get_kc_result_filename(args, get_default_fold(args))) as kc_file:
            kc_info = json.load(kc_file)

    # visualize_single(kc_info)
    visualize_average(kc_info, args)
