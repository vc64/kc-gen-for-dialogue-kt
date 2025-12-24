from typing import Dict, List
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers import SentenceTransformer

from dialogue_kt.models.dkt_sem import ALT_ARCH
from dialogue_kt.prompting import kt_system_prompt, kt_user_prompt, dkt_sem_prompt
from dialogue_kt.utils import device, to_device

def apply_annotations(sample: dict, apply_na: bool = True):
    dialogue = sample["dialogue"]
    anno = sample["annotation"]
    if "error" in anno:
        return None
    # Handle dialogues beginning with turn 0 (student-initiated)
    if dialogue[0]["turn"] == 0:
        anno["turn 0"] = {"correct": None, "kcs": []}
    # Copy correctness and kcs into dialogue
    for dia_turn in dialogue:
        anno_turn = anno[f"turn {dia_turn['turn']}"]
        corr = anno_turn["correct"]
        kcs = anno_turn["kcs"]
        if apply_na:
            corr = None if not kcs else corr
            kcs = [] if corr is None else kcs
        dia_turn["correct"] = dia_turn["og_correct"] = corr
        dia_turn["kcs"] = kcs
    # Use human annotation of correctness for final turn
    if dialogue[-1]["kcs"]: # Skip if no KCs for final turn since correct must be None
        if "expected_result" in sample["meta_data"]: # CoMTA
            dialogue[-1]["correct"] = sample["meta_data"]["expected_result"] == "Answer Accepted"
        elif "self_correctness" in sample["meta_data"]: # MathDial
            if dialogue[-1]["correct"] is not None: # Final turn could be closing remarks, so skip if not tagged as having correctness
                if sample["meta_data"]["self_correctness"] == "Yes":
                    dialogue[-1]["correct"] = True
                elif sample["meta_data"]["self_correctness"] == "Yes, but I had to reveal the answer":
                    dialogue[-1]["correct"] = None
                elif sample["meta_data"]["self_correctness"] == "No":
                    dialogue[-1]["correct"] = False
    return dialogue

class DatasetBase(Dataset):
    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class LMKTDatasetUnpacked(DatasetBase):
    def __init__(self, data: pd.DataFrame, tokenizer, args, skip_first_turn: bool = False):
        self.data = []
        failed = 0
        for idx, sample in data.iterrows():
            dialogue = apply_annotations(sample)
            if not dialogue:
                failed += 1
                continue
            is_first_turn = True
            for turn in dialogue:
                if turn["correct"] is None:
                    continue
                # Skip first tagged turn at test time for fairness with baselines
                if skip_first_turn and is_first_turn:
                    is_first_turn = False
                    continue
                self.data.append({
                    "dialogue_idx": idx,
                    "prompts": [
                        tokenizer.apply_chat_template([
                            {"role": "system", "content": kt_system_prompt(args)},
                            {"role": "user", "content": kt_user_prompt(sample, dialogue, turn["turn"], kc, args)},
                            {"role": "assistant", "content": f"\n"} # Newline would precede True or False prediction
                        ], tokenize=False)
                        for kc in turn["kcs"]
                    ],
                    "label": turn["correct"],
                    "kcs": turn["kcs"]
                })
        print(f"{failed} / {len(data)} dialogues failed processing")
        print(f"Number of data points: {len(self.data)}")

class LMKTCollatorUnpacked:
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, batch):
        all_prompts = [prompt for sample in batch for prompt in sample["prompts"]]
        prompts_tokenized = self.tokenizer(all_prompts, return_tensors="pt", padding=True)
        prompts_tokenized = to_device(prompts_tokenized, device, self.args)
        return {
            "input_ids": prompts_tokenized.input_ids,
            "attention_mask": prompts_tokenized.attention_mask,
            "last_idxs": prompts_tokenized.attention_mask.sum(dim=-1) - 2, # Take index of token before eos
            "num_kcs": to_device(torch.LongTensor([len(sample["prompts"]) for sample in batch]), device, self.args),
            "labels": to_device(torch.Tensor([sample["label"] for sample in batch]), device, self.args),
            "meta_data": batch
        }

class LMKTDatasetPacked(DatasetBase):
    def __init__(self, data: pd.DataFrame, tokenizer, args, skip_first_turn: bool = False):
        self.data = []
        failed = 0
        for idx, sample in data.iterrows():
            dialogue = apply_annotations(sample)
            if not dialogue:
                failed += 1
                continue
            is_first_turn = True
            for turn in dialogue:
                if turn["correct"] is None:
                    continue
                # Skip first tagged turn at test time for fairness with baselines
                if skip_first_turn and is_first_turn:
                    is_first_turn = False
                    continue
                # Create base prompt followed by all possible continuations
                prompt = tokenizer.apply_chat_template([
                    {"role": "system", "content": kt_system_prompt(args)},
                    {"role": "user", "content": kt_user_prompt(sample, dialogue, turn["turn"], None, args)},
                ], tokenize=False)
                kc_conts = [
                    tokenizer.apply_chat_template([
                        {"role": "user", "content": kc},
                        {"role": "assistant", "content": f"\n"} # Newline would precede True or False prediction
                    ], tokenize=False)
                    for kc in turn["kcs"]
                ]
                kc_conts = [" " + cont.split("user<|end_header_id|>\n\n")[1] for cont in kc_conts]
                prompt = prompt + "".join(kc_conts)
                self.data.append({
                    "dialogue_idx": idx,
                    "prompt": prompt,
                    "label": turn["correct"],
                    "kcs": turn["kcs"]
                })
        print(f"{failed} / {len(data)} dialogues failed processing")
        print(f"Number of data points: {len(self.data)}")
        #prompts = [sample["prompt"] for sample in self.data]
        #prompts_tokenized = tokenizer(prompts, return_tensors="pt", padding=True)
        #lengths = [len(x) for x in prompts_tokenized["input_ids"]]
        #print(f"Max tokens per prompt: {max(lengths)}")
        #print(f"Total tokens: {sum(lengths)}")

class LMKTCollatorPacked:
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, batch):
        prompts = [sample["prompt"] for sample in batch]
        prompts_tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = to_device(prompts_tokenized.input_ids, device, self.args)
        batch_size, max_seq_len = input_ids.shape
        eos_idxs = [
            (input_ids[seq_idx] == self.tokenizer.eos_token_id).nonzero().squeeze().cpu()
            for seq_idx in range(batch_size)
        ]
        # Create default lower triangular 3D attention mask
        attention_mask = torch.ones((max_seq_len, max_seq_len)).tril().repeat(batch_size, 1, 1)
        tril_mask = attention_mask[0].type(torch.bool)
        # Create default 2D position id matrix
        position_ids = torch.arange(max_seq_len).repeat(batch_size, 1)
        # Set attention mask and position ids for each sequence
        for seq_idx in range(batch_size):
            # Get end of context
            context_end_idx = eos_idxs[seq_idx][1]
            # Initialize to no attention to any tokens after context
            attention_mask[seq_idx, :, position_ids[seq_idx] >= context_end_idx] = 0
            # Update attention mask and position ids for each KC
            start_idx = context_end_idx + 1
            for end_idx in eos_idxs[seq_idx][3::2]:
                # Set position ids as if KC immediately followed context
                position_ids[seq_idx, start_idx : end_idx + 1] = torch.arange(context_end_idx, context_end_idx + end_idx - start_idx + 1)
                # Set KC attention mask to lower triangular to permit self-attention
                cur_tril_mask = tril_mask.clone()
                cur_tril_mask[end_idx + 1:] = False
                cur_tril_mask[:, :start_idx] = False
                attention_mask[seq_idx, cur_tril_mask] = 1
                # Go to next KC
                start_idx = end_idx + 1

        # Get index of token before eos for each KC, pad for easier loss computation
        last_idxs = pad_sequence([idxs[3::2] - 1 for idxs in eos_idxs], batch_first=True)
        return {
            "input_ids": input_ids,
            "attention_mask": to_device(attention_mask.unsqueeze(1), device, self.args), # Add singleton head dimension
            "position_ids": to_device(position_ids, device, self.args),
            "last_idxs": last_idxs,
            "num_kcs": to_device(torch.LongTensor([len(sample["kcs"]) for sample in batch]), device, self.args),
            "labels": to_device(torch.Tensor([sample["label"] for sample in batch]), device, self.args),
            "meta_data": batch
        }

class DKTDataset(DatasetBase):
    def __init__(self, data: pd.DataFrame, kc_dict: Dict[str, int], kc_emb_matrix: torch.Tensor, sbert_model: SentenceTransformer):
        self.data = []
        failed = 0
        num_data_points = 0
        num_correct = 0
        for idx, sample in data.iterrows():
            dialogue = apply_annotations(sample)
            if not dialogue:
                failed += 1
                continue
            dialogue_data = {
                "labels": [], "labels_flat": [], "kc_ids": [], "kc_ids_flat": [], "turn_end_idxs": [],
                "teacher_turns": [], "student_turns": [], "kcs": [], "kc_embs": [],
                "dialogue": dialogue, "dialogue_idx": idx
            }
            for turn in dialogue:
                if turn["correct"] is None:
                    continue
                dialogue_data["labels"].append(turn["correct"])
                dialogue_data["kc_ids"].append([kc_dict[kc] for kc in turn["kcs"]])
                for kc in turn["kcs"]:
                    dialogue_data["labels_flat"].append(turn["correct"])
                    dialogue_data["kc_ids_flat"].append(kc_dict[kc])
                dialogue_data["turn_end_idxs"].append(len(dialogue_data["kc_ids_flat"]) - 1)
                dialogue_data["teacher_turns"].append(turn["teacher"])
                dialogue_data["student_turns"].append(turn["student"])
                dialogue_data["kcs"].append(turn["kcs"])
                if kc_emb_matrix is not None:
                    dialogue_data["kc_embs"].append(
                        kc_emb_matrix[dialogue_data["kc_ids"][-1]].mean(dim=0)
                    )
            # Add dialogue if at least 2 turns tagged, otherwise nothing to predict
            if len(dialogue_data["labels"]) > 1:
                self.data.append(dialogue_data)
                num_data_points += len(dialogue_data["labels"])
                num_correct += sum(dialogue_data["labels"])
        # Batch encode all dialogue text
        if sbert_model is not None:
            batch_size = 512
            if ALT_ARCH:
                seqs = [dkt_sem_prompt(tt, st, kcs, corr)
                        for dialogue in self.data
                        for tt, st, kcs, corr in zip(dialogue["teacher_turns"], dialogue["student_turns"], dialogue["kcs"], dialogue["labels"])]
                result_embs = []
                for batch_start_idx in range(0, len(seqs), batch_size):
                    batch = seqs[batch_start_idx : batch_start_idx + batch_size]
                    result_embs.append(sbert_model.encode(batch, convert_to_tensor=True))
                result_embs = torch.concat(result_embs, dim=0)
                turn_counter = 0
                for dialogue in self.data:
                    seq_len = len(dialogue["labels"])
                    dialogue["turn_embs"] = result_embs[turn_counter : turn_counter + seq_len]
                    turn_counter += seq_len
            else:
                seqs = [turn for dialogue in self.data for turn in dialogue["teacher_turns"]] + [
                        turn for dialogue in self.data for turn in dialogue["student_turns"]]
                result_embs = []
                for batch_start_idx in range(0, len(seqs), batch_size):
                    batch = seqs[batch_start_idx : batch_start_idx + batch_size]
                    result_embs.append(sbert_model.encode(batch, convert_to_tensor=True))
                result_embs = torch.concat(result_embs, dim=0)
                turn_counter = 0
                for dialogue in self.data:
                    seq_len = len(dialogue["labels"])
                    dialogue["teacher_embs"] = result_embs[turn_counter : turn_counter + seq_len]
                    stud_start = result_embs.shape[0] // 2
                    dialogue["student_embs"] = result_embs[stud_start + turn_counter : stud_start + turn_counter + seq_len]
                    turn_counter += seq_len
        self.majority_class = 1 if num_correct / num_data_points >= .5 else 0
        print(f"{failed} / {len(data)} dialogues failed processing")
        print(f"Num dialogues: {len(self.data)}, num data points: {num_data_points}, {num_correct} correct")

class DKTCollator:
    def __init__(self, flatten_kcs: bool):
        self.flatten_kcs = flatten_kcs

    def __call__(self, batch):
        labels = pad_sequence(
            [torch.LongTensor(seq["labels"]) for seq in batch],
            batch_first=True, padding_value=-100 # Pad with -100 to ignore loss on padding regions
        )
        # # Fill in KC ids, 2D matrix (length x max num KCs) per sequence
        num_kcs = pad_sequence(
            [torch.LongTensor([len(kc_ids) for kc_ids in seq["kc_ids"]]) for seq in batch],
            batch_first=True, padding_value=1 # Pad with 1 to avoid division by 0
        )
        max_num_kcs = num_kcs.max()
        kc_ids = torch.zeros((*num_kcs.shape, max_num_kcs), dtype=torch.long)
        for seq_idx, seq in enumerate(batch):
            for turn_idx, turn_kc_ids in enumerate(seq["kc_ids"]):
                kc_ids[seq_idx, turn_idx, :len(turn_kc_ids)] = torch.LongTensor(turn_kc_ids)

        result = {
            "labels": labels.to(device),
            "kc_ids": kc_ids.to(device),
            "num_kcs": num_kcs.to(device)
        }

        if self.flatten_kcs:
            # Add flattened versions of KC ids and labels for unrolled model input
            kc_ids_flat = pad_sequence([torch.LongTensor(seq["kc_ids_flat"]) for seq in batch], batch_first=True)
            labels_flat = pad_sequence([torch.LongTensor(seq["labels_flat"]) for seq in batch], batch_first=True)
            turn_end_idxs = pad_sequence([torch.LongTensor(seq["turn_end_idxs"]) for seq in batch], batch_first=True)
            result = {
                **result,
                "labels_flat": labels_flat.to(device),
                "kc_ids_flat": kc_ids_flat.to(device),
                "turn_end_idxs": turn_end_idxs.to(device)
            }

        # Add text embeddings for DKT-Sem
        if batch[0]["kc_embs"] and not ALT_ARCH:
            kc_embs = pad_sequence([torch.stack(seq["kc_embs"]) for seq in batch], batch_first=True)
            teacher_embs = pad_sequence([seq["teacher_embs"] for seq in batch], batch_first=True)
            student_embs = pad_sequence([seq["student_embs"] for seq in batch], batch_first=True)
            result = {
                **result,
                "kc_embs": kc_embs,
                "teacher_embs": teacher_embs,
                "student_embs": student_embs
            }
        elif "turn_embs" in batch[0]:
            turn_embs = pad_sequence([seq["turn_embs"] for seq in batch], batch_first=True)
            result = {
                **result,
                "turn_embs": turn_embs
            }

        return result

def get_dataloader(dataset: Dataset, collator, batch_size: int, shuffle: bool):
    return DataLoader(dataset, collate_fn=collator, batch_size=batch_size, shuffle=shuffle)
    
    # if not batch_len_sampler:
    #     return DataLoader(dataset, collate_fn=collator, batch_size=batch_size, shuffle=shuffle)
    
    # def approx_len(example):
    #     return len(example["prompt"])

    # lengths = [approx_len(x) for x in dataset.data]
    # sampler = LengthGroupedSampler(
    #     batch_size=batch_size,
    #     lengths=lengths,
    # )
    # return DataLoader(dataset, collate_fn=collator, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
