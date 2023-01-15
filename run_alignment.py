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
from esm.esm import pretrained
from src import Embedding
from dataset import SCOPePairsDataset, LSTMDataset, SSADataset
from Bio import SeqIO
import logging
import datetime
from pandas import *
from utils import (
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
    result = {}
    if args.eval_dataset == 'bb3_release':
        fasta_dir = Path("../../data/bb3_release")
    elif args.eval_dataset == 'homfam-small':
        fasta_dir = Path("../../data/homfam/small")
    elif args.eval_dataset == 'homfam-medium':
        fasta_dir = Path("../../data/homfam/medium")
    elif args.eval_dataset == 'homfam-large':
        fasta_dir = Path("../../data/homfam/large")
    elif args.eval_dataset == 'exthomfam-small':
        fasta_dir = Path("../../data/extHomFam-v3/small")
    elif args.eval_dataset == 'exthomfam-medium':
        fasta_dir = Path("../../data/extHomFam-v3/medium")
    elif args.eval_dataset == 'exthomfam-huge':
        fasta_dir = Path("../../data/extHomFam-v3/huge")
    elif args.eval_dataset == 'exthomfam-large':
        fasta_dir = Path("../../data/extHomFam-v3/large")
    elif args.eval_dataset == 'exthomfam-xlarge':
        fasta_dir = Path("../../data/extHomFam-v3/xlarge")
    elif args.eval_dataset == 'oxfam-small':
        fasta_dir = Path("../../data/oxfam/small")
    elif args.eval_dataset == 'oxfam-medium':
        fasta_dir = Path("../../data/oxfam/medium")
    elif args.eval_dataset == 'oxfam-large':
        fasta_dir = Path("../../data/oxfam/large")
    elif args.eval_dataset == 'ContTest-small':
        fasta_dir = Path("../../data/ContTest/data/small")
    elif args.eval_dataset == 'ContTest-medium':
        fasta_dir = Path("../../data/ContTest/data/medium")
    elif args.eval_dataset == 'ContTest-large':
        fasta_dir = Path("../../data/ContTest/data/large")
    else:
        raise NotImplementedError
    if 'ContTest' in args.eval_dataset:
        fasta_files = []
        for path in fasta_dir.iterdir():
            if path.is_dir() and "PF" in path.stem:
                prefix = path.stem.split('_')[0]
                fasta_files.append(path / f"{prefix}_unaligned.fasta")
    else:
        fasta_files = list(fasta_dir.glob('**/*.tfa'))
    for i, fastaFile in enumerate(tqdm(fasta_files, desc="Eval Guide Tree")):
        # print(f"Now processing file ({i + 1}/{len(list(fasta_dir.glob('**/*.tfa')))}) : {fastaFile.name}")
        logging.info(f"Now processing file ({i + 1}/{len(fasta_files)}) : {fastaFile.name}")
        raw_seqs = list(SeqIO.parse(fastaFile, 'fasta'))
        ## Alignment
        if args.align_prog == "clustalo":
            if args.eval_dataset == 'bb3_release':
                msf_path = args.msf_dir / f"{fastaFile.stem}.msf"
                runcmd(f"./clustalo --threads=8 --outfmt=msf --in {fastaFile.absolute().resolve()} --out {msf_path.absolute().resolve()} --force")
            else:
                pfa_path = fastaFile.parent / f"{fastaFile.stem.split('_')[0]}_clustalo.fasta" if 'ContTest' in args.eval_dataset else args.msf_dir / f"{fastaFile.stem}.pfa"
                runcmd(f"./clustalo --threads=8 --in {fastaFile.absolute().resolve()} --out {pfa_path.absolute().resolve()} --force")
        elif args.align_prog == "mafft":
            if args.eval_dataset == 'bb3_release':
                raise NotImplementedError
            pfa_path = fastaFile.parent / f"{fastaFile.stem.split('_')[0]}_mafft.fasta" if 'ContTest' in args.eval_dataset else args.msf_dir / f"{fastaFile.stem}.pfa"
            ret = runcmd(f"./mafft --large --anysymbol --thread 8 {fastaFile.absolute().resolve()}").decode().split('\n')
            with open(pfa_path, 'w') as f:
                for line in ret:
                    f.write(line + '\n')
        else:
            if args.eval_dataset == 'bb3_release':
                raise NotImplementedError
            if args.align_prog == "famsa":
                pfa_path = fastaFile.parent / f"{fastaFile.stem.split('_')[0]}_famsa.fasta" if 'ContTest' in args.eval_dataset else args.msf_dir / f"{fastaFile.stem}.pfa"
                runcmd(f"famsa -keep-duplicates -gt upgma -t 8 {fastaFile.absolute().resolve()} {pfa_path.absolute().resolve()}")
            else:
                pfa_path = fastaFile.parent / f"{fastaFile.stem.split('_')[0]}_tcoffee.fasta" if 'ContTest' in args.eval_dataset else args.msf_dir / f"{fastaFile.stem}.pfa"
                runcmd(f"t_coffee -reg -thread 8 -child_thread 8 -seq {fastaFile.absolute().resolve()} -nseq {min(200, len(raw_seqs) // 10)} -tree mbed -method mafftginsi_msa -outfile {pfa_path.absolute().resolve()}")
        ## Calculate Score
        if 'ContTest' in args.eval_dataset:
            continue
        if args.eval_dataset == 'bb3_release':
            xml_path = fastaFile.parents[0] / f"{fastaFile.stem}.xml"
            output = runcmd(f"bali_score {xml_path} {msf_path}").decode("utf-8").split('\n')[10]
            SP = float(output.split()[2])
            TC = float(output.split()[3])
        else:
            rfa_path = list(fasta_dir.glob(f"{fastaFile.stem}.rfa"))[0]
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
        # print(f"SP-score = {SP}")
        # print(f"TC = {TC}")
        logging.info(f"SP-score = {SP}")
        logging.info(f"TC = {TC}")
        # Collect Score
        category = fastaFile.parents[0].name
        if category not in result:
            result[category] = {
                'SP' : [SP],
                'TC' : [TC]
            }
        else:
            result[category]['SP'].append(SP)
            result[category]['TC'].append(TC)
    
    final_result = {}
    if 'ContTest' in args.eval_dataset:
        # runcmd(f"cd {fasta_dir.absolute().resolve()}")
        # runcmd(f"runbenchmark -a {args.align_prog}")
        # csv = read_csv(f"results/results_{args.align_prog}_psicov.csv")
        # PFAM_ID = csv['PFAM_ID'].tolist()
        # PROTEIN_ID = csv[' PROTEIN_ID'].tolist()
        # SCORE = csv[' SCORE'].tolist()
        # total_score = 0
        # for pfam, prot, score in zip(PFAM_ID, PROTEIN_ID, SCORE):
        #     logging.info(f"{pfam}\t\t{prot}\t\t{score}")
        #     total_score += float(score)
        cat = fasta_dir.stem
        final_result[cat] = {
            "SCORE" : 0
        }
    else:
        for cat, value in result.items():
            final_result[cat] = {
                "SP" : sum(value['SP']) / len(value['SP']),
                "TC" : sum(value['TC']) / len(value['TC']),
            }
    return final_result

def eval_Kmeans(model, args):
    result = {}
    model = model.cuda()
    if args.ref is not None:
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
        #### Release Memory ####
        del(raw_seqs)
        del(seqs)
        gc.collect()
        ########################
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
            batch_size=128,
            collate_fn=dataset.collate_fn,
            pin_memory=True,
            # batch_sampler=dataset.batch_sampler(args.toks_per_batch_eval),
            shuffle=False,
            num_workers=8,
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
        # print(f"Finish embedding in {time.time() - start_time} secs.")
        logging.info(f"Finish embedding in {time.time() - start_time} secs.")
        
        centers, clusters = BisectingKmeans(unique_sorted_seqs)
        # print(f"Cluster Sizes : {[len(cl) for cl in clusters]}")
        logging.info(f"Cluster Sizes : {[len(cl) for cl in clusters]}")
        if len(centers) > 1:
            center_embeddings = torch.stack([cen['embedding'] for cen in centers], dim=0)
            ## Create Distance Matrix
            dist_matrix = torch.cdist(center_embeddings, center_embeddings, 2).fill_diagonal_(1000000000).cpu().numpy() / 50
            # dist_matrix = L2_distance(center_embeddings) / 50
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
            if args.eval_dataset == 'bb3_release':
                msf_path = args.msf_dir / f"{fastaFile.stem}.msf"
                runcmd(f"./clustalo --threads=8 --outfmt=msf --in {fastaFile.absolute().resolve()} --out {msf_path.absolute().resolve()} --guidetree-in {tree_path.absolute().resolve()} --force")
            else:
                pfa_path = fastaFile.parent / f"{fastaFile.stem.split('_')[0]}_clustaloNN.fasta" if 'ContTest' in args.eval_dataset else args.msf_dir / f"{fastaFile.stem}.pfa"
                runcmd(f"./clustalo --threads=8 --in {fastaFile.absolute().resolve()} --out {pfa_path.absolute().resolve()} --guidetree-in {tree_path.absolute().resolve()} --force")
        elif args.align_prog == "mafft":
            if args.eval_dataset == 'bb3_release':
                raise NotImplementedError
            mafft_path = args.tree_dir / f"{fastaFile.stem}_mafft.dnd"
            pfa_path = fastaFile.parent / f"{fastaFile.stem.split('_')[0]}_mafftNN.fasta" if 'ContTest' in args.eval_dataset else args.msf_dir / f"{fastaFile.stem}.pfa"
            ret = runcmd(f"./newick2mafft.rb {tree_path.absolute().resolve()}").decode().split('\n')
            with open(mafft_path, 'w') as f:
                for line in ret:
                    f.write(line + '\n')
            ret = runcmd(f"./mafft --anysymbol --thread 8 --treein {mafft_path.absolute().resolve()} {fastaFile.absolute().resolve()}").decode().split('\n')
            with open(pfa_path, 'w') as f:
                for line in ret:
                    f.write(line + '\n')
        else:
            if args.eval_dataset == 'bb3_release':
                raise NotImplementedError
            
            if args.align_prog == "famsa":
                pfa_path = fastaFile.parent / f"{fastaFile.stem.split('_')[0]}_famsaNN.fasta" if 'ContTest' in args.eval_dataset else args.msf_dir / f"{fastaFile.stem}.pfa"
                runcmd(f"famsa -keep-duplicates -t 8 -gt import {tree_path.absolute().resolve()} {fastaFile.absolute().resolve()} {pfa_path.absolute().resolve()}")
            else:
                pfa_path = fastaFile.parent / f"{fastaFile.stem.split('_')[0]}_tcoffeeNN.fasta" if 'ContTest' in args.eval_dataset else args.msf_dir / f"{fastaFile.stem}.pfa"
                runcmd(f"t_coffee -reg -thread 8 -child_thread 8 -seq {fastaFile.absolute().resolve()} -nseq {min(200, num_seqs // 10)} -tree {tree_path.absolute().resolve()} -method mafftgins1_msa -outfile {pfa_path.absolute().resolve()}")

        ## Calculate Score
        if 'ContTest' in args.eval_dataset:
            continue
        if args.eval_dataset == 'bb3_release':
            xml_path = fastaFile.parents[0] / f"{fastaFile.stem}.xml"
            output = runcmd(f"bali_score {xml_path} {msf_path}").decode("utf-8").split('\n')[10]
            SP = float(output.split()[2])
            TC = float(output.split()[3])
        else:
            rfa_path = list(fasta_dir.glob(f"{fastaFile.stem}.rfa"))[0]
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
        # print(f"SP-score = {SP}")
        # print(f"TC = {TC}")
        logging.info(f"SP-score = {SP}")
        logging.info(f"TC = {TC}")
        # Collect Score
        category = fastaFile.parents[0].name
        if category not in result:
            result[category] = {
                'SP' : [SP],
                'TC' : [TC]
            }
        else:
            result[category]['SP'].append(SP)
            result[category]['TC'].append(TC)
    
    final_result = {}
    if 'ContTest' in args.eval_dataset:
        # runcmd(f"cd {fasta_dir.absolute().resolve()}")
        # runcmd(f"runbenchmark -a {args.align_prog}")
        # csv = read_csv(f"results/results_{args.align_prog}_psicov.csv")
        # PFAM_ID = csv['PFAM_ID'].tolist()
        # PROTEIN_ID = csv[' PROTEIN_ID'].tolist()
        # SCORE = csv[' SCORE'].tolist()
        # total_score = 0
        # for pfam, prot, score in zip(PFAM_ID, PROTEIN_ID, SCORE):
        #     logging.info(f"{pfam}\t\t{prot}\t\t{score}")
        #     total_score += float(score)
        cat = fasta_dir.stem
        final_result[cat] = {
            "SCORE" : 0
        }
    else:
        for cat, value in result.items():
            final_result[cat] = {
                "SP" : sum(value['SP']) / len(value['SP']),
                "TC" : sum(value['TC']) / len(value['TC']),
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
    
    logging.info(f"========== Before Training ==========")
    # print(f"Evaluation Loss : {eval_loss}")
    # print(f"Evaluation Spearman Correlation : {eval_spearman}")
    # print(f"Evaluation Pearson Correlation : {eval_pearson}")
    logging.info(f"Guide Tree Evaluation : ")
    if "ContTest" in args.eval_dataset:
        logging.info(f"Category\t\tSCORE")
        for key, value in result.items():
            logging.info(f"{key}\t\t{value['SCORE']}")
    else:
        logging.info(f"Category\t\tSP\t\tTC")
        for key, value in result.items():
            logging.info(f"{key}\t\t{value['SP']}\t\t{value['TC']}")
    logging.info(f"Total Execution Time : {time.time() - tot_start_time} (s)")
    logging.info(f"===================================")
    

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
        "--lstm_path",
        type=Path,
        help="Path to pretrained LSTM model.",
        default="../../ckpt/prose/saved_models/prose_mt_3x1024.sav",
    )
    parser.add_argument(
        "--esm_path",
        type=str,
        help="Path to pretrained esm model.",
        default="./esm/ckpt/esm2_t33_650M_UR50D.pt",
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
    parser.add_argument('--embed_type', type=str, default='LSTM', choices=['LSTM', 'esm-43M', 'esm-35M', 'esm-150M', 'esm-650M'])
    
    # eval
    # parser.add_argument("--eval_tree_dir", type=Path, default="../../data/bb3_release")
    parser.add_argument("--toks_per_batch_eval", type=int, default=16384)
    parser.add_argument("--newick2mafft_path", type=Path, default="./newick2mafft.rb")
    parser.add_argument("--fastSP_path", type=Path, default="./FastSP/FastSP.jar")
    parser.add_argument("--align_prog", type=str, default='clustalo', choices=["clustalo", "mafft", "famsa", "tcoffee"])
    ## WARNING : Do not use LCS if sequences are not unique !!
    parser.add_argument("--dist_type", type=str, default="NW", choices=["NW", "SW", "LCS"])
    parser.add_argument("--eval_dataset", type=str, default="bb3_release", choices=[
        "bb3_release", 
        "homfam-small", "homfam-medium", "homfam-large", 
        "oxfam-small", "oxfam-medium", "oxfam-large", 
        "exthomfam-small", "exthomfam-medium", "exthomfam-large", "exthomfam-huge", "exthomfam-xlarge",
        "ContTest-small", "ContTest-medium", "ContTest-large", 
    ])
    parser.add_argument("--no_tree", action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # sys.stdout = open('./logging.txt', "w")
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    args.tree_dir.mkdir(parents=True, exist_ok=True)
    args.msf_dir.mkdir(parents=True, exist_ok=True)
    args.fasta_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.no_tree:
        log_filename = args.log_dir / datetime.datetime.now().strftime(f"{args.align_prog}_{args.eval_dataset}_%Y-%m-%d_%H_%M_%S.log")
    else:
        log_filename = args.log_dir / datetime.datetime.now().strftime(f"mix_{args.align_prog}_{args.eval_dataset}_%Y-%m-%d_%H_%M_%S.log")        
    logging.basicConfig(
        level=logging.INFO, 
        filename=log_filename, 
        filemode='w',
	    format='[%(asctime)s %(levelname)-8s] %(message)s',
	    datefmt='%Y%m%d %H:%M:%S',
	)
    main(args)