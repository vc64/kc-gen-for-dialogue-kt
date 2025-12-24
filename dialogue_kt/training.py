import json
from tqdm import tqdm
import torch
import transformers
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from pykt.models.dkt import DKT
from pykt.models.akt import AKT
from pykt.models.dkvmn import DKVMN
from pykt.models.saint import SAINT
from pyBKT.models import Model as BKT
from sentence_transformers import SentenceTransformer

from dialogue_kt.models.lm import get_model
from dialogue_kt.models.dkt_multi_kc import DKTMultiKC
from dialogue_kt.models.dkt_sem import DKTSem
from dialogue_kt.models.simplekt import simpleKT
from dialogue_kt.data_loading import (load_annotated_data, get_kc_result_filename, get_qual_result_filename, get_default_fold, load_kc_dict,
                          correct_to_str, standards_to_str, get_model_file_suffix, COMTA_SUBJECTS)
from dialogue_kt.kt_data_loading import (LMKTDatasetUnpacked, LMKTCollatorUnpacked, LMKTDatasetPacked, LMKTCollatorPacked,
                             DKTDataset, DKTCollator, get_dataloader, apply_annotations)
from dialogue_kt.prompting import get_true_false_tokens
from dialogue_kt.utils import device, get_checkpoint_path

# ===== Common Functions =====

def apply_defaults(args):
    if args.model_type == "lmkt":
        defaults = {
            "epochs": 3,
            "lr": 2e-4,
            "wd": 1e-2,
            "gc": 1.0,
            "batch_size": 1,
            "grad_accum_steps": 64,
            "r": 16,
            "lora_alpha": 16
        }
    else:
        defaults = {
            "epochs": 100,
            "lr": 1e-3,
            "wd": 1e-2,
            "gc": 0,
            "batch_size": 64,
            "grad_accum_steps": 1,
            "emb_size": 64
        }
    for key, val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)
    print("Arguments:", args)

def hyperparam_sweep(args):
    apply_defaults(args)
    args.testonval = True
    args.crossval = args.dataset == "comta"
    model_names = []
    results = []
    if args.model_type == "lmkt":
        for lr in [5e-5, 1e-4, 2e-4, 3e-4]:
            for r in [4, 8, 16, 32]:
                args.model_name = f"hpsweep_{args.dataset}_{args.tag_src}_lmkt_agg{args.agg}_lr{lr}_r{r}"
                args.lr = lr
                args.r = r
                args.lora_alpha = r
                model_names.append(args.model_name)
                results.append(train(args))
    else:
        for lr in [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]:
            for emb_size in [8, 16, 32, 64, 128, 256, 512]:
                args.model_name = f"hpsweep_{args.dataset}_{args.tag_src}_{args.model_type}_agg{args.agg}_lr{lr}_es{emb_size}"
                args.lr = lr
                args.emb_size = emb_size
                model_names.append(args.model_name)
                results.append(train(args))
    aucs = np.array([metrics.mean(0)[2] if args.crossval else metrics[2] for metrics in results])
    best_model_idx = aucs.argmax()
    result_str = "\n".join([f"{model_name}: {auc:.2f}" for model_name, auc in zip(model_names, aucs)])
    result_str += f"\nBest: {model_names[best_model_idx]}: {aucs[best_model_idx]:.2f}"
    print(result_str)
    with open(f"results/metrics_hpsweep_{args.dataset}_{args.tag_src}_{args.model_type}_agg{args.agg}.txt", "w") as file:
        file.write(result_str + "\n")

def crossval(args, fn):
    # Train/test models across folds
    metrics_agg = []
    folds = COMTA_SUBJECTS if args.split_by_subject else range(1, 6)
    for fold in folds:
        print(f"Fold {fold}...")
        metrics = fn(args, fold)
        metrics_agg.append(metrics)
    # Aggregate and report metrics across folds
    metrics_np = np.stack(metrics_agg, axis=0)
    avg = metrics_np.mean(axis=0)
    std = metrics_np.std(axis=0)
    metric_names = ["Loss", "Acc", "AUC", "Prec", "Rec", "F1"]
    if len(avg) > 6:
        metric_names += ["Acc (Final)", "AUC (Final)", "Prec (Final)", "Rec (Final)", "F1 (Final)"]
    results = [
        f"{metric}: ${avg[idx]:.2f}_{{\\pm {std[idx]:.2f}}}$" for idx, metric in
        enumerate(metric_names)
    ]
    result_str = "\n".join(results)
    print(result_str)
    with open(f"results/metrics_crossval{'_subj' if args.split_by_subject else ''}_{get_model_file_suffix(args)}.txt", "w") as out_file:
        out_file.writelines([
            str(metrics_agg) + "\n",
            result_str + "\n"
        ])
    # Aggregate and save qual analysis files across folds
    if args.model_type == "lmkt":
        dfs = [pd.read_csv(get_qual_result_filename(args, fold)) for fold in folds]
        pd.concat(dfs).to_csv(get_qual_result_filename(args), index=False)
    return metrics_np

def train(args):
    if args.hyperparam_sweep:
        args.hyperparam_sweep = False
        return hyperparam_sweep(args)

    assert args.model_name or args.model_type == "bkt"
    apply_defaults(args)
    fn = train_lmkt if args.model_type == "lmkt" else train_test_bkt if args.model_type == "bkt" else train_baseline
    if args.crossval:
        return crossval(args, fn)
    else:
        return fn(args, get_default_fold(args))

def test(args):
    apply_defaults(args)
    fn = test_lmkt if args.model_type == "lmkt" else test_baseline
    if args.crossval:
        return crossval(args, fn)
    else:
        return fn(args, get_default_fold(args))

def compute_metrics(labels, preds):
    hard_preds = np.round(preds)
    acc = accuracy_score(labels, hard_preds)
    auc = roc_auc_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, hard_preds, average="binary")
    return acc * 100, auc * 100, prec * 100, rec * 100, f1 * 100

def compute_all_metrics(loss, all_labels, all_preds, final_turn_labels, final_turn_preds, args, fold):
    result_str = f"Loss: {loss:.4f}\n"
    result_str += f"Overall ({len(all_labels)} samples):\n"
    result_str += f"GT - True: {sum(all_labels)}, False: {len(all_labels) - sum(all_labels)}; "
    result_str += f"Pred - True: {sum(np.round(all_preds))}, False: {len(all_preds) - sum(np.round(all_preds))}\n"
    all_metrics = compute_metrics(all_labels, all_preds)
    result_str += "Acc: {:.2f}, AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}\n".format(*all_metrics)
    if final_turn_labels is not None:
        result_str += f"Final Turn ({len(final_turn_labels)} samples):\n"
        result_str += f"GT - True: {sum(final_turn_labels)}, False: {len(final_turn_labels) - sum(final_turn_labels)}; "
        result_str += f"Pred - True: {sum(np.round(final_turn_preds))}, False: {len(final_turn_preds) - sum(np.round(final_turn_preds))}\n"
        final_metrics = compute_metrics(final_turn_labels, final_turn_preds)
        result_str += "Acc: {:.2f}, AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}\n".format(*final_metrics)
    else:
        final_metrics = []
    print(result_str)
    with open(f"results/metrics_{get_model_file_suffix(args, fold)}.txt", "w") as out_file:
        out_file.write(result_str)
    return all_metrics, final_metrics


# ===== LMKT =====

def get_lmkt_loss_unpacked(model, batch, true_token, false_token, args):
    # Get logits at last token of each sequence
    model_output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    batch_size = model_output.logits.shape[0]
    logits = model_output.logits[torch.arange(batch_size), batch["last_idxs"]]
    # Return probability of True token over False token for each sequence
    logits = torch.stack([logits[:, true_token], logits[:, false_token]], dim=1)
    kc_probs = torch.softmax(logits, dim=1)[torch.arange(batch_size), 0]
    # Get probability that all KCs are True for each turn in the batch
    num_kc_counter = 0
    kc_probs_grouped = []
    corr_probs = []
    for num_kcs in batch["num_kcs"]:
        kc_probs_grouped.append(kc_probs[num_kc_counter : num_kc_counter + num_kcs].tolist())
        if args.agg == "prod":
            prob = kc_probs[num_kc_counter : num_kc_counter + num_kcs].prod()
        elif args.agg == "mean-ar":
            prob = kc_probs[num_kc_counter : num_kc_counter + num_kcs].mean()
        elif args.agg == "mean-geo":
            prob = kc_probs[num_kc_counter : num_kc_counter + num_kcs].prod() ** (1 / num_kcs)
        corr_probs.append(prob)
        num_kc_counter += num_kcs
    corr_probs = torch.stack(corr_probs)
    # Get BCE loss with correctness labels and predicted probabilities
    loss = torch.nn.BCELoss()(corr_probs, batch["labels"])
    return loss, kc_probs_grouped, corr_probs

def get_lmkt_loss_packed(model, batch, true_token, false_token, args):
    # Invert attention mask
    attention_mask = batch["attention_mask"]
    min_dtype = torch.finfo(model.dtype).min
    attention_mask[attention_mask == 0] = min_dtype
    attention_mask[attention_mask == 1] = 0
    attention_mask = attention_mask.type(model.dtype)
    # Get logits at last token of each sequence
    model_output = model(input_ids=batch["input_ids"], attention_mask=attention_mask, position_ids=batch["position_ids"])
    batch_size = model_output.logits.shape[0]
    logits = model_output.logits[torch.arange(batch_size).unsqueeze(1), batch["last_idxs"]]
    # Return probability of True token over False token for each sequence
    logits = torch.stack([logits[:, :, true_token], logits[:, :, false_token]], dim=2)
    kc_probs = torch.softmax(logits, dim=2)[:, :, 0]
    # Get probability that all KCs are True for each turn in the batch
    kc_probs_grouped = [probs[:num_kcs].tolist() for probs, num_kcs in zip(kc_probs, batch["num_kcs"])]
    # Set probs on padded indices
    padding_val = 0 if args.agg == "mean-ar" else 1
    kc_probs = torch.masked_scatter(kc_probs, batch["last_idxs"].to(device) == 0, torch.full_like(kc_probs, padding_val).to(device))
    # Get BCE loss with correctness labels and predicted probabilities
    if args.agg == "prod":
        corr_probs = kc_probs.prod(dim=1)
    elif args.agg == "mean-ar":
        corr_probs = kc_probs.sum(dim=1) / batch["num_kcs"]
    elif args.agg == "mean-geo":
        corr_probs = kc_probs.prod(dim=1) ** (1 / batch["num_kcs"])
    loss = torch.nn.BCELoss()(corr_probs, batch["labels"])
    return loss, kc_probs_grouped, corr_probs

def train_lmkt(args, fold):
    # Load language model with trainable LoRA adapters
    model, tokenizer = get_model(args.base_model, False, pt_model_name=args.pt_model_name, r=args.r, lora_alpha=args.lora_alpha, quantize=args.quantize, use_gradient_checkpointing=True)
    model.print_trainable_parameters()

    # Load and split dataset, annotated with correctness and KCs
    KTDataset = LMKTDatasetPacked if args.pack_kcs else LMKTDatasetUnpacked
    KTCollator = LMKTCollatorPacked if args.pack_kcs else LMKTCollatorUnpacked
    get_loss = get_lmkt_loss_packed if args.pack_kcs else get_lmkt_loss_unpacked
    train_df, val_df, _ = load_annotated_data(args, fold)
    if args.debug:
        train_df = train_df[:2]
        val_df = val_df[:2]
        print(train_df.iloc[0])
        print(val_df.iloc[0])
    train_dataset = KTDataset(train_df, tokenizer, args)
    val_dataset = KTDataset(val_df, tokenizer, args)
    collator = KTCollator(tokenizer, args)
    train_dataloader = get_dataloader(train_dataset, collator, args.batch_size, True)
    val_dataloader = get_dataloader(val_dataset, collator, args.batch_size, False)

    # For finding logits for loss
    true_token, false_token = get_true_false_tokens(tokenizer)

    # Do training loop
    if args.optim == "adamw":
        #optimizer = SOAP(model.parameters(), lr = args.lr, betas=(.95, .95), weight_decay=args.wd, precondition_frequency=10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = transformers.Adafactor(model.parameters(), lr=args.lr, weight_decay=args.wd, relative_step=False)
    best_val_loss = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}")
        total_train_loss = 0
        total_val_loss = 0

        model.train()
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            loss, _, _ = get_loss(model, batch, true_token, false_token, args)
            total_train_loss += loss.item()
            loss = loss / args.grad_accum_steps
            loss.backward()
            if (batch_idx + 1) % args.grad_accum_steps == 0 or batch_idx == len(train_dataloader) - 1:
                if args.gc:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gc)
                optimizer.step()
                optimizer.zero_grad()

        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_dataloader, desc="Validating"):
                loss, _, _ = get_loss(model, batch, true_token, false_token, args)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if not best_val_loss or avg_val_loss < best_val_loss:
            print("Best! Saving model...")
            model_name = args.model_name + (f"_{fold}" if fold else "")
            model.save_pretrained(get_checkpoint_path(model_name))
            best_val_loss = avg_val_loss

    return test_lmkt(args, fold)

def test_lmkt(args, fold):
    # Load trained language model
    model_name = args.model_name and args.model_name + (f"_{fold}" if fold else "")
    model, tokenizer = get_model(args.base_model, True, model_name=model_name, quantize=args.quantize)
    model.eval()

    # Load annotated data
    KTDataset = LMKTDatasetPacked if args.pack_kcs else LMKTDatasetUnpacked
    KTCollator = LMKTCollatorPacked if args.pack_kcs else LMKTCollatorUnpacked
    get_loss = get_lmkt_loss_packed if args.pack_kcs else get_lmkt_loss_unpacked
    _, val_df, test_df = load_annotated_data(args, fold)
    if args.testonval:
        test_df = val_df
    if args.debug:
        test_df = test_df[:10]
        print(test_df.iloc[0])
    test_dataset = KTDataset(test_df, tokenizer, args, skip_first_turn=not args.inc_first_label)
    collator = KTCollator(tokenizer, args)
    test_dataloader = get_dataloader(test_dataset, collator, args.batch_size, False)

    # For finding logits for loss
    true_token, false_token = get_true_false_tokens(tokenizer)

    # Collect meta data and predicted KC/correctness probabilities for test set
    dialogue_idx_to_sample_idxs = {}
    all_labels = []
    all_preds = []
    all_kc_probs = []
    all_kcs = []
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(test_dataloader)):
        for sample_idx, sample in enumerate(batch["meta_data"]):
            dialogue_idx_to_sample_idxs.setdefault(sample["dialogue_idx"], []).append(batch_idx + sample_idx)
        with torch.no_grad():
            loss, kc_probs, corr_probs = get_loss(model, batch, true_token, false_token, args)
        total_loss += loss.item()
        all_labels.extend(batch["labels"].tolist())
        all_preds.extend(corr_probs.tolist())
        all_kc_probs.extend(kc_probs)
        all_kcs.extend([sample["kcs"] for sample in batch["meta_data"]])

    # Compute quantitative metrics and save metrics file
    loss = total_loss / len(test_dataloader)
    final_turn_labels = [all_labels[idxs[-1]] for idxs in dialogue_idx_to_sample_idxs.values()]
    final_turn_preds = [all_preds[idxs[-1]] for idxs in dialogue_idx_to_sample_idxs.values()]
    all_metrics, final_metrics = compute_all_metrics(loss, all_labels, all_preds, final_turn_labels, final_turn_preds, args, fold)

    # Save file for visual analysis
    kc_results = {
        dialogue_idx: [
            {
                kc: kc_prob
                for kc, kc_prob in zip(all_kcs[sample_idx], all_kc_probs[sample_idx])
            }
            for sample_idx in sample_idxs
        ]
        for dialogue_idx, sample_idxs in dialogue_idx_to_sample_idxs.items()
    }
    with open(get_kc_result_filename(args, fold), "w") as out_file:
        json.dump(kc_results, out_file, indent=2)

    # Save file for qualitative analysis
    qual_data = []
    for dia_idx, sample in test_df.iterrows():
        dialogue = apply_annotations(sample)
        if dia_idx not in dialogue_idx_to_sample_idxs:
            continue
        dia_preds = [all_preds[idx] for idx in dialogue_idx_to_sample_idxs[dia_idx]]
        dia_labels = [all_labels[idx] for idx in dialogue_idx_to_sample_idxs[dia_idx]]
        dia_acc = f"{(np.round(dia_preds) == dia_labels).mean():.4f}"
        first_turn = not args.inc_first_label
        label_counter = 0
        for turn in dialogue:
            if turn["correct"] is not None and not first_turn:
                label_idx = dialogue_idx_to_sample_idxs[dia_idx][label_counter]
                prob = f"{all_preds[label_idx]:.4f}"
                kc_probs = ", ".join([f"{kc_prob:.4f}" for kc_prob in all_kc_probs[label_idx]])
                label_counter += 1
            else:
                prob = "--"
                kc_probs = "--"
            if turn["correct"] is not None and first_turn:
                first_turn = False
            qual_data.append({
                "Dialogue ID": dia_idx,
                "Turn": turn["turn"],
                "Teacher": turn["teacher"] or "--",
                "Student": turn["student"],
                "Correct": correct_to_str(turn["correct"]),
                "Prob": prob,
                "KC Probs": kc_probs,
                "Dialogue Acc.": dia_acc,
                "KCs": standards_to_str(turn["kcs"], "\n"),
                "Notes": ""
            })
        qual_data.append({key: "" for key in qual_data[0]})
    pd.DataFrame(qual_data).to_csv(get_qual_result_filename(args, fold), index=False)

    return np.array([loss, *all_metrics, *final_metrics])


# ===== Baselines =====

BASELINE_MODELS = ["dkt-multi", "dkt-sem", "dkt", "akt", "dkvmn", "saint", "simplekt"]
NON_FLAT_KC_ARCH = ["dkt-multi", "dkt-sem"]

def select_flat_baseline_out_vectors(y: torch.Tensor, batch, shift_turn_end_idxs: bool):
    if shift_turn_end_idxs:
        # Predict KCs with output from first KC of turn for models where correctness is only visible in previous idxs
        # Clip at end to prevent out of bounds, no effect since last pred unused
        batch["turn_end_idxs"] = torch.clip(batch["turn_end_idxs"] + 1, max=batch["turn_end_idxs"].max())
    # Get output vectors at index of last KC per turn (to predict next turn's KCs)
    turn_end_idxs = batch["turn_end_idxs"].unsqueeze(2).repeat(1, 1, y.shape[2])
    return torch.gather(y, 1, turn_end_idxs)

def get_baseline_loss(y: torch.Tensor, batch, args):
    # Aggregate KC probs from outputs, one output per question
    batch_size, max_seq_len, max_num_kcs = batch["kc_ids"].shape
    kc_pad_mask = torch.arange(max_num_kcs).repeat(batch_size, max_seq_len, 1).to(device) >= batch["num_kcs"].unsqueeze(2)
    y = y[:, :-1].contiguous() # Last item in sequence doesn't predict anything
    kc_probs = torch.gather(y, 2, batch["kc_ids"][:, 1:]) # Collect KC predictions for next question, B x L x K
    # Set probs on padded indices
    padding_val = 0 if args.agg == "mean-ar" else 1
    kc_probs = torch.masked_scatter(kc_probs, kc_pad_mask[:, 1:], torch.full_like(kc_probs, padding_val).to(device))
    # Calculate correct probabilities (B x L)
    if args.agg == "prod":
        corr_probs = kc_probs.prod(dim=2)
    elif args.agg == "mean-ar":
        corr_probs = kc_probs.sum(dim=2) / batch["num_kcs"][:, 1:]
    elif args.agg == "mean-geo":
        corr_probs = kc_probs.prod(dim=2) ** (1 / batch["num_kcs"][:, 1:])

    # Compute BCE loss
    labels_flat = batch["labels"][:, 1:].contiguous().view(-1)
    loss_mask = labels_flat != -100
    labels_flat = labels_flat[loss_mask].type(torch.float)
    corr_probs_flat = corr_probs.view(-1)[loss_mask]
    loss: torch.Tensor = torch.nn.BCELoss()(corr_probs_flat, labels_flat)
    return loss, corr_probs

def get_baseline_model(kc_dict: dict, kc_emb_matrix: torch.Tensor, args):
    num_kcs = len(kc_dict)
    emb_size = args.emb_size
    n_blocks = 4 # For layered models
    if args.model_type == "dkt-multi":
        return DKTMultiKC(num_kcs, emb_size).to(device)
    if args.model_type == "dkt-sem":
        return DKTSem(emb_size, kc_emb_matrix).to(device)
    if args.model_type == "dkt":
        return DKT(num_kcs, emb_size).to(device)
    if args.model_type == "akt":
        model = AKT(num_kcs, num_kcs, emb_size, n_blocks, 0.05, emb_size, final_fc_dim=emb_size)
        model.out[3] = torch.nn.Linear(emb_size, emb_size) # Reduce from 256 to emb_size to avoid overparameterization
        model.out[6] = torch.nn.Linear(emb_size, num_kcs) # Predict all KCs instead of just current question
        return model.to(device)
    if args.model_type == "dkvmn":
        model = DKVMN(num_kcs, emb_size, 50)
        model.p_layer = torch.nn.Linear(emb_size, num_kcs) # Predict all KCs instead of just current question
        return model.to(device)
    if args.model_type == "saint":
        model = SAINT(num_kcs, num_kcs, 256, emb_size, 8, 0.2, n_blocks)
        model.out = torch.nn.Linear(emb_size, num_kcs) # Predict all KCs instead of just current question
        return model.to(device)
    if args.model_type == "simplekt":
        model = simpleKT(num_kcs, num_kcs, emb_size, n_blocks, 0.2, d_ff=emb_size, final_fc_dim=emb_size, final_fc_dim2=emb_size)
        model.out[6] = torch.nn.Linear(emb_size, num_kcs) # Predict all KCs instead of just current question
        return model.to(device)
    raise Exception(f"Model {args.model_type} not supported")

def compute_baseline_loss(model, batch, args):
    if args.model_type == "dkt-multi":
        y = model(batch)
        return get_baseline_loss(y, batch, args)
    elif args.model_type == "dkt-sem":
        y = model(batch)
        return get_baseline_loss(y, batch, args)
    elif args.model_type == "dkt":
        y = model(batch["kc_ids_flat"], batch["labels_flat"])
        y = select_flat_baseline_out_vectors(y, batch, False)
        return get_baseline_loss(y, batch, args)
    elif args.model_type == "akt":
        y, rasch_loss = model(batch["kc_ids_flat"], batch["labels_flat"], batch["kc_ids_flat"])
        y = select_flat_baseline_out_vectors(y, batch, True)
        loss, corr_probs = get_baseline_loss(y, batch, args)
        loss += rasch_loss
        return loss, corr_probs
    elif args.model_type == "dkvmn":
        y = model(batch["kc_ids_flat"], batch["labels_flat"])
        y = select_flat_baseline_out_vectors(y, batch, True)
        return get_baseline_loss(y, batch, args)
    elif args.model_type == "saint":
        y = model(batch["kc_ids_flat"], batch["kc_ids_flat"], batch["labels_flat"][:, :-1])
        y = select_flat_baseline_out_vectors(y, batch, True)
        return get_baseline_loss(y, batch, args)
    elif args.model_type == "simplekt":
        y = model({
            "qseqs": batch["kc_ids_flat"][:, :-1],
            "cseqs": batch["kc_ids_flat"][:, :-1],
            "rseqs": batch["labels_flat"][:, :-1],
            "shft_qseqs": batch["kc_ids_flat"][:, 1:],
            "shft_cseqs": batch["kc_ids_flat"][:, 1:],
            "shft_rseqs": batch["labels_flat"][:, 1:]
        })
        y = select_flat_baseline_out_vectors(y, batch, True)
        return get_baseline_loss(y, batch, args)
    raise Exception(f"Model {args.model_type} not supported")

def compute_kc_emb_matrix(sbert_model: SentenceTransformer, kc_dict: dict):
    print("Computing SBERT embeddings...")
    kcs = [kv[0] for kv in sorted(kc_dict.items(), key=lambda kv: kv[1])]
    kc_emb_matrix = sbert_model.encode(kcs, convert_to_tensor=True)
    return kc_emb_matrix

def train_baseline(args, fold):
    assert args.model_type in BASELINE_MODELS

    # Load KC dictionary and optionally text embeddings
    kc_dict = load_kc_dict(args)
    if args.model_type == "dkt-sem":
        sbert_model = SentenceTransformer("all-mpnet-base-v2")
        kc_emb_matrix = compute_kc_emb_matrix(sbert_model, kc_dict)
    else:
        sbert_model = None
        kc_emb_matrix = None

    # Create model
    model = get_baseline_model(kc_dict, kc_emb_matrix, args)

    # Load and split dataset, annotated with correctness and KCs
    train_df, val_df, _ = load_annotated_data(args, fold)
    if args.debug:
        train_df = train_df[:2]
        val_df = val_df[:2]
        print(train_df.iloc[0])
        print(val_df.iloc[0])
    flatten_kcs = args.model_type not in NON_FLAT_KC_ARCH # Flatten KCs in sequence for architectures that don't support multi-KCs
    train_dataset = DKTDataset(train_df, kc_dict, kc_emb_matrix, sbert_model)
    val_dataset = DKTDataset(val_df, kc_dict, kc_emb_matrix, sbert_model)
    collator = DKTCollator(flatten_kcs)
    train_dataloader = get_dataloader(train_dataset, collator, args.batch_size, True)
    val_dataloader = get_dataloader(val_dataset, collator, args.batch_size, False)

    # Do training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val_loss = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}")
        total_train_loss = 0
        total_val_loss = 0

        model.train()
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            loss, _ = compute_baseline_loss(model, batch, args)
            total_train_loss += loss.item()
            loss = loss / args.grad_accum_steps
            loss.backward()
            if (batch_idx + 1) % args.grad_accum_steps == 0 or batch_idx == len(train_dataloader) - 1:
                if args.gc:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gc)
                optimizer.step()
                optimizer.zero_grad()

        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_dataloader, desc="Validating"):
                loss, _ = compute_baseline_loss(model, batch, args)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if not best_val_loss or avg_val_loss < best_val_loss:
            print("Best! Saving model...")
            model_name = args.model_name + (f"_{fold}" if fold else "") + ".pt"
            torch.save(model.state_dict(), get_checkpoint_path(model_name))
            best_val_loss = avg_val_loss

    return test_baseline(args, fold)

def test_baseline(args, fold):
    # Load KC dictionary and optionally text embeddings
    kc_dict = load_kc_dict(args)
    if args.model_type == "dkt-sem":
        sbert_model = SentenceTransformer("all-mpnet-base-v2")
        kc_emb_matrix = compute_kc_emb_matrix(sbert_model, kc_dict)
    else:
        sbert_model = None
        kc_emb_matrix = None

    # Load trained model
    if args.model_type in BASELINE_MODELS:
        model = get_baseline_model(kc_dict, kc_emb_matrix, args)
        model_name = args.model_name + (f"_{fold}" if fold else "") + ".pt"
        model.load_state_dict(torch.load(get_checkpoint_path(model_name), map_location=device))
        model.eval()
    else:
        model = None

    # Load annotated data
    _, val_df, test_df = load_annotated_data(args, fold)
    if args.testonval:
        test_df = val_df
    if args.debug:
        test_df = test_df[:10]
        print(test_df.iloc[0])
    flatten_kcs = args.model_type not in NON_FLAT_KC_ARCH # Flatten KCs in sequence for architectures that don't support multi-KCs
    test_dataset = DKTDataset(test_df, kc_dict, kc_emb_matrix, sbert_model)
    collator = DKTCollator(flatten_kcs)
    test_dataloader = get_dataloader(test_dataset, collator, args.batch_size, False)

    # Collect meta data and predicted KC/correctness probabilities for test set
    all_labels = []
    all_preds = []
    final_turn_labels = []
    final_turn_preds = []
    total_loss = 0
    for batch in tqdm(test_dataloader):
        labels = batch["labels"][:, 1:]
        if model is not None:
            with torch.no_grad():
                loss, corr_probs = compute_baseline_loss(model, batch, args)
        elif args.model_type == "random":
            corr_probs = torch.zeros_like(labels).random_(0, 2)
            loss = torch.tensor(0)
        elif args.model_type == "majority":
            corr_probs = torch.full_like(labels, fill_value=test_dataset.majority_class)
            loss = torch.tensor(0)
        total_loss += loss.item()
        mask = labels != -100
        all_labels.extend(labels[mask].tolist())
        all_preds.extend(corr_probs[mask].tolist())
        final_idxs = mask.sum(dim=1) - 1
        final_turn_labels.extend(labels[torch.arange(mask.shape[0]), final_idxs].tolist())
        final_turn_preds.extend(corr_probs[torch.arange(mask.shape[0]), final_idxs].tolist())

    # Compute quantitative metrics across all turns and only on final turns
    loss = total_loss / len(test_dataloader)
    all_metrics, final_metrics = compute_all_metrics(loss, all_labels, all_preds, final_turn_labels, final_turn_preds, args, fold)

    return np.array([loss, *all_metrics, *final_metrics])

def bkt_prep_data(df: pd.DataFrame, kc_dict: dict):
    dataset = DKTDataset(df, kc_dict, None, None)
    results = []
    order_id = 0
    for _, sample in enumerate(dataset.data):
        for kc, label in zip(sample["kc_ids_flat"], sample["labels_flat"]):
            results.append({"user_id": sample["dialogue_idx"], "skill_name": str(kc), "correct": label, "order_id": order_id})
            order_id += 1
    return pd.DataFrame(results), dataset

def train_test_bkt(args, fold):
    # Load and reformat data
    kc_dict = load_kc_dict(args)
    train_df, val_df, test_df = load_annotated_data(args, fold)
    train_df, _ = bkt_prep_data(pd.concat([train_df, val_df]), kc_dict)
    test_df, test_dataset = bkt_prep_data(test_df, kc_dict)

    # Train model
    model = BKT(seed=221, num_fits=1)
    model.fit(data=train_df)

    # Test model
    print("Train Acc./AUC:", model.evaluate(data=train_df, metric=["accuracy", "auc"]))
    print("Test Acc./AUC:", model.evaluate(data=test_df, metric=["accuracy", "auc"]))
    pred_df: pd.DataFrame = model.predict(data=test_df)
    pred_df = pred_df.sort_values(["order_id"])
    all_labels = []
    all_preds = []
    for sample, (_, user) in zip(test_dataset, pred_df.groupby("user_id", sort=False)):
        preds_flat = user["correct_predictions"]
        labels = sample["labels"]
        preds = []
        prev_idx = 0
        for turn_end_idx in sample["turn_end_idxs"]:
            if args.agg == "prod":
                preds.append(np.prod(preds_flat[prev_idx : turn_end_idx + 1]))
            elif args.agg == "mean-ar":
                preds.append(np.mean(preds_flat[prev_idx : turn_end_idx + 1]))
            elif args.agg == "mean-geo":
                preds.append(np.prod(preds_flat[prev_idx : turn_end_idx + 1]) ** (1 / (turn_end_idx - prev_idx + 1)))
            prev_idx = turn_end_idx + 1
        all_labels.extend(labels[1:])
        all_preds.extend(preds[1:])
    metrics, _ = compute_all_metrics(0, all_labels, all_preds, None, None, args, fold)
    return [0, *metrics]
