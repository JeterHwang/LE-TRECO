import json
import os
import random
import re
import numpy as np
import math
import gc
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import time 
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
from src import Embedding
from src import LSTMDataset
from Bio import SeqIO
import logging
import datetime
from pandas import *
from src import (
    L2_distance, 
    SSA_score_slow, 
    UPGMA, UPGMA_Kmeans,
    runcmd, 
    calculate_corr, 
    BisectingKmeans
)

def same_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True

def eval_prog(args):
    result = {
        'SP' : [],
        'TC' : []
    }
    model = model.cuda()
    if args.ref is not None:
        assert args.ref.is_dir() or args.ref.is_file()
        assert (args.ref.is_dir() and args.input.is_dir()) or (args.ref.is_file() and args.input.is_file())
    
    if args.input.is_dir():
        fasta_files = list(args.input.glob('**/*.tfa'))
    elif args.input.is_file():
        fasta_files = [args.input]
    else:
        raise NotImplementedError
    
    for i, fastaFile in enumerate(tqdm(fasta_files, desc="Eval Guide Tree")):
        # print(f"Now processing file ({i + 1}/{len(list(fasta_dir.glob('**/*.tfa')))}) : {fastaFile.name}")
        logging.info(f"Now processing file ({i + 1}/{len(fasta_files)}) : {fastaFile.name}")
        raw_seqs = list(SeqIO.parse(fastaFile, 'fasta'))
        ## Alignment
        if args.align_prog == "clustalo":
            pfa_path = args.msf_dir / f"{fastaFile.stem}.pfa"
            runcmd(f"./clustalo --threads={args.thread} --in {fastaFile.absolute().resolve()} --out {pfa_path.absolute().resolve()} --force")
        elif args.align_prog == "mafft":
            pfa_path = args.msf_dir / f"{fastaFile.stem}.pfa"
            ret = runcmd(f"./mafft --large --anysymbol --thread {args.thread} {fastaFile.absolute().resolve()}").decode().split('\n')
            with open(pfa_path, 'w') as f:
                for line in ret:
                    f.write(line + '\n')
        elif args.align_prog == "famsa":
            pfa_path = args.msf_dir / f"{fastaFile.stem}.pfa"
            runcmd(f"famsa -keep-duplicates -gt upgma -t {args.thread} {fastaFile.absolute().resolve()} {pfa_path.absolute().resolve()}")
        else:
            pfa_path = args.msf_dir / f"{fastaFile.stem}.pfa"
            runcmd(f"t_coffee -reg -thread {args.thread} -child_thread {args.thread} -seq {fastaFile.absolute().resolve()} -nseq {min(200, len(raw_seqs) // 10)} -tree mbed -method mafftginsi_msa -outfile {pfa_path.absolute().resolve()}")
        
        ## Calculate Score
        if args.ref.is_dir():
            rfa_path = list(args.ref.glob(f"{fastaFile.stem}.rfa"))[0]
        else:
            rfa_path = args.ref

        rfa_pfa_path = args.msf_dir / f"{fastaFile.stem}_rfa.pfa"
        rfa_raw = list(SeqIO.parse(rfa_path, 'fasta'))
        pfa_raw = list(SeqIO.parse(pfa_path, 'fasta'))
        seq_in_ref = [str(ss.id) for ss in rfa_raw]
        with open(rfa_pfa_path, 'w') as f:
            for pfa in pfa_raw:
                seq_name = str(pfa.id)
                seq_data = str(pfa.seq)
                if seq_name in seq_in_ref:
                    f.write(f">{seq_name}\n")
                    f.write(f"{seq_data}\n")
        raw_scores = runcmd(f"java -jar {args.fastSP_path.absolute().resolve()} -r {rfa_path.absolute().resolve()} -e {rfa_pfa_path.absolute().resolve()}").decode().split()
        SP = float(raw_scores[raw_scores.index('SP-Score') + 1])
        TC = float(raw_scores[raw_scores.index('TC') + 1])
        
        logging.info(f"SP-score = {SP}")
        logging.info(f"TC = {TC}")
        
        # Collect Score
        result['SP'].append(SP)
        result['TC'].append(TC)

    
    final_result = {
        'SP' : sum(result['SP']) / len(result['SP']),
        'TC' : sum(result['TC']) / len(result['TC'])
    }
    return final_result

def eval_Kmeans(model, args):
    result = {
        'SP' : [],
        'TC' : []
    }
    model = model.cuda()
    if args.ref is not None:
        assert args.ref.is_dir() or args.ref.is_file()
        assert (args.ref.is_dir() and args.input.is_dir()) or (args.ref.is_file() and args.input.is_file())
    
    if args.input.is_dir():
        fasta_files = list(args.input.glob('**/*.tfa'))
    elif args.input.is_file():
        fasta_files = [args.input]
    else:
        raise NotImplementedError

    for i, fastaFile in enumerate(tqdm(fasta_files, desc="Eval Guide Tree")):
        # print(f"Now processing file ({i + 1}/{len(list(fasta_dir.glob('**/*.tfa')))}) : {fastaFile.name}")
        logging.info(f"Now processing file ({i + 1}/{len(fasta_files)}) : {fastaFile.name}")
        ## Read sequences
        raw_seqs = list(SeqIO.parse(fastaFile, 'fasta'))
        seqs, avg_len, num_seqs = [], 0, len(raw_seqs)
        for idx, seq in enumerate(raw_seqs):
            seqs.append({
                'num' : idx + 1,
                'name' : str(idx + 1) if args.align_prog == 'mafft' else str(seq.id),
                'seq' : str(seq.seq),
                'embedding' : None,
            })
            avg_len += len(str(seq.seq))
        avg_len /= len(raw_seqs)
        # print(f"Average Sequence Length : {avg_len}")
        logging.info(f"Average Sequence Length : {avg_len}")
        std = 0
        for idx, seq in enumerate(raw_seqs):
            std += (len(str(seq.seq)) - avg_len)**2
        std = math.sqrt(std / len(raw_seqs))
        logging.info(f"Standard Deviationo of Sequence Length : {std}")
        ##
        sorted_seqs = sorted(seqs, key=lambda seq : len(seq['seq']), reverse=True)
        ##### Release Memory #####
        del(raw_seqs)           ##
        del(seqs)               ##
        gc.collect()            ##
        ##########################
        queue, id2cluster = [], {}
        for i, seq in enumerate(sorted_seqs):
            if i > 0 and sorted_seqs[i]['seq'] == sorted_seqs[i-1]['seq']:
                queue[-1].append(sorted_seqs[i])
            else:
                queue.append([sorted_seqs[i]])
        unique_sorted_seqs = []
        for uniq in queue:
            unique_sorted_seqs.append(uniq[0])
            if len(uniq) > 1:
                id2cluster[uniq[0]['name']] = uniq
            else:
                id2cluster[uniq[0]['name']] = None
        unique_sorted_seqs.sort(key=lambda seq : seq['num'])
        # print(f"Unique sequences : {len(unique_sorted_seqs)} / {len(seqs)}")
        logging.info(f"Unique sequences : {len(unique_sorted_seqs)} / {num_seqs}")
        
        ## Create Dataset / Dataloader
        if model.alphabet is not None:
            dataset = LSTMDataset([seq['seq'] for seq in unique_sorted_seqs], model.alphabet)
        else:
            dataset = LSTMDataset([seq['seq'] for seq in unique_sorted_seqs])
        eval_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=dataset.collate_fn,
            pin_memory=True,
            # batch_sampler=dataset.batch_sampler(args.toks_per_batch_eval),
            shuffle=False,
            num_workers=args.num_workers,
        )
        ## Create Embeddings
        embeddings, index = [], []
        start_time = time.time()
        with torch.no_grad():
            for i, (tokens, lengths, indices) in enumerate(tqdm(eval_loader, desc="Embed Sequences:")):
                tokens = tokens.cuda()
                emb = model(tokens, lengths)
                embeddings.append(emb.cpu())
        embeddings = torch.cat(embeddings, dim=0)
        for idx, emb in enumerate(embeddings):
            unique_sorted_seqs[idx]['embedding'] = emb
        logging.info(f"Finish embedding in {time.time() - start_time} secs.")
        
        centers, clusters = BisectingKmeans(unique_sorted_seqs, args.min_cluster_size)
        logging.info(f"Cluster Sizes : {[len(cl) for cl in clusters]}")
        if len(centers) > 1:
            center_embeddings = torch.stack([cen['embedding'] for cen in centers], dim=0)
            ## Create Distance Matrix
            dist_matrix = torch.cdist(center_embeddings, center_embeddings, 2).fill_diagonal_(1000000000).cpu().numpy() / 50
        else:
            center_embeddings = None
            dist_matrix = None

        ## UPGMA / Output Guide Tree
        tree_path = args.tree_dir / f"{fastaFile.stem}.dnd"
        UPGMA_Kmeans(dist_matrix, clusters, id2cluster, tree_path, args.fasta_dir, args.dist_type)
        ####### Release Memory ######
        del embeddings              #
        del center_embeddings       #
        del unique_sorted_seqs      #
        del queue                   #
        del clusters                #
        del centers                 #
        del id2cluster              #
        gc.collect()                #
        #############################
        ## Alignment
        if args.align_prog == "clustalo":
            pfa_path = args.msf_dir / f"{fastaFile.stem}.pfa"
            runcmd(f"./clustalo --threads={args.thread} --in {fastaFile.absolute().resolve()} --out {pfa_path.absolute().resolve()} --guidetree-in {tree_path.absolute().resolve()} --force")
        elif args.align_prog == "mafft":
            mafft_path = args.tree_dir / f"{fastaFile.stem}_mafft.dnd"
            pfa_path = args.msf_dir / f"{fastaFile.stem}.pfa"
            ret = runcmd(f"./newick2mafft.rb {tree_path.absolute().resolve()}").decode().split('\n')
            with open(mafft_path, 'w') as f:
                for line in ret:
                    f.write(line + '\n')
            ret = runcmd(f"./mafft --anysymbol --thread {args.thread} --treein {mafft_path.absolute().resolve()} {fastaFile.absolute().resolve()}").decode().split('\n')
            with open(pfa_path, 'w') as f:
                for line in ret:
                    f.write(line + '\n')
        elif args.align_prog == "famsa":
            pfa_path = args.msf_dir / f"{fastaFile.stem}.pfa"
            runcmd(f"famsa -keep-duplicates -t {args.thread} -gt import {tree_path.absolute().resolve()} {fastaFile.absolute().resolve()} {pfa_path.absolute().resolve()}")
        else:
            pfa_path = fastaFile.parent / f"{fastaFile.stem.split('_')[0]}_tcoffeeNN.fasta" if 'ContTest' in args.eval_dataset else args.msf_dir / f"{fastaFile.stem}.pfa"
            runcmd(f"t_coffee -reg -thread {args.thread} -child_thread {args.thread} -seq {fastaFile.absolute().resolve()} -nseq {min(200, num_seqs // 10)} -tree {tree_path.absolute().resolve()} -method mafftgins1_msa -outfile {pfa_path.absolute().resolve()}")

        ## Calculate Score
        if args.ref.is_dir():
            rfa_path = list(args.ref.glob(f"{fastaFile.stem}.rfa"))[0]
        else:
            rfa_path = args.ref

        rfa_pfa_path = args.msf_dir / f"{fastaFile.stem}_rfa.pfa"
        rfa_raw = list(SeqIO.parse(rfa_path, 'fasta'))
        pfa_raw = list(SeqIO.parse(pfa_path, 'fasta'))
        seq_in_ref = [str(ss.id) for ss in rfa_raw]
        with open(rfa_pfa_path, 'w') as f:
            for pfa in pfa_raw:
                seq_name = str(pfa.id)
                seq_data = str(pfa.seq)
                if seq_name in seq_in_ref:
                    f.write(f">{seq_name}\n")
                    f.write(f"{seq_data}\n")
        raw_scores = runcmd(f"java -jar {args.fastSP_path.absolute().resolve()} -r {rfa_path.absolute().resolve()} -e {rfa_pfa_path.absolute().resolve()}").decode().split()
        SP = float(raw_scores[raw_scores.index('SP-Score') + 1])
        TC = float(raw_scores[raw_scores.index('TC') + 1])
        
        logging.info(f"SP-score = {SP}")
        logging.info(f"TC = {TC}")
        
        # Collect Score
        result['SP'].append(SP)
        result['TC'].append(TC)
    
    final_result = {
        'SP' : sum(result['SP']) / len(result['SP']),
        'TC' : sum(result['TC']) / len(result['TC'])
    }
    return final_result

def main(args):
    same_seed(args.seed)

    model = Embedding.load_pretrained(
        args.embed_type, 
        args.model_path,
    )
    
    tot_start_time = time.time()
    if args.no_tree:
        result = eval_prog(args)
    else: 
        result = eval_Kmeans(
            model,
            args
        )
    
    logging.info(f"============== Guide Tree Evaluation ==============")
    logging.info(f"SP\t\tTC")
    logging.info(f"{round(result['SP'] * 100, 1)}\t\t{round(result['TC'] * 100, 1)}")        
    logging.info(f"Total Execution Time : {time.time() - tot_start_time} (s)")
    logging.info(f"===================================================")
    

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        help="Directory to the dataset.",
        default="./data/homfam/small",
    )
    parser.add_argument(
        "--ref",
        type=Path,
        help="Path to the reference alignment",
        default=None
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        help="Path to pretrained model.",
        default="../../ckpt/prose/saved_models/prose_mt_3x1024.sav",
    )
    parser.add_argument(
        "--tree_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./trees",
    )
    parser.add_argument(
        "--msf_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./msf",
    )
    parser.add_argument(
        "--fasta_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./fasta",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        help="Path to logs",
        default='./logs'
    )

    # training
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--thread", type=int, default=8)
    parser.add_argument('--embed_type', type=str, default='LSTM', choices=['LSTM', 'esm-43M', 'esm-35M', 'esm-150M', 'esm-650M'])
    parser.add_argument("--min_cluster_size", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    # eval
    parser.add_argument("--toks_per_batch_eval", type=int, default=16384)
    parser.add_argument("--newick2mafft_path", type=Path, default="./newick2mafft.rb")
    parser.add_argument("--fastSP_path", type=Path, default="./FastSP-1.7.1/FastSP.jar")
    parser.add_argument("--align_prog", type=str, default='clustalo', choices=["clustalo", "mafft", "famsa", "tcoffee"])

    parser.add_argument("--dist_type", type=str, default="NW", choices=["NW", "SW"])
    parser.add_argument("--no_tree", action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.tree_dir.mkdir(parents=True, exist_ok=True)
    args.msf_dir.mkdir(parents=True, exist_ok=True)
    args.fasta_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.no_tree:
        log_filename = args.log_dir / datetime.datetime.now().strftime(f"{args.align_prog}_{args.embed_type}_{args.dist_type}_%Y-%m-%d_%H_%M_%S.log")
    else:
        log_filename = args.log_dir / datetime.datetime.now().strftime(f"mix_{args.align_prog}_{args.embed_type}_{args.dist_type}_%Y-%m-%d_%H_%M_%S.log")        
    logging.basicConfig(
        level=logging.INFO, 
        filename=log_filename, 
        filemode='w',
	    format='[%(asctime)s %(levelname)-8s] %(message)s',
	    datefmt='%Y%m%d %H:%M:%S',
	)
    main(args)