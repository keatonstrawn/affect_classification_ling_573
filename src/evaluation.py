#!/usr/bin/env python
import pandas as pd
import sys
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import io


class Evaluator:


    def __init__(self, input_dir, output_dir):
            """Generates features from processed data to be used in hate speech detection tasks A and B, as specified in
            SemEval 2019 task 5.

            Includes methods to generate the following features:
                * _NRC_counts --> binary classification of words across ten emotional dimensions. Raw counts are then
                                        transformed into proportions to normalize across tweets.
                * example feature2 --> fill this in with actual feature
                * example feature3 --> fill this in with actual feature
            """
            
            # Initialize the directories
            self.input_dir = input_dir
            self.output_dir = output_dir
           


    def evaluate_a(self,pred,gold):
        levels = ["HS"]

        ground_truth = pd.read_csv(gold, sep="\t", names=["ID", "Tweet-text", "HS", "TargetRange", "Aggressiveness"],
                                converters={0: str, 1: str, 2: int, 3: int, 4: int}, header=None, skiprows=1)

        predicted = pd.read_csv(pred, sep="\t", names=["ID"] + levels ,
                                converters={0: str, 1: int}, header=None, skiprows=1)

        # Check length files
        if (len(ground_truth) != len(predicted)):
            sys.exit('Prediction and gold data have different number of lines.')

        # Check predicted classes
        for c in levels:
            gt_class = list(ground_truth[c].value_counts().keys())
            if not (predicted[c].isin(gt_class).all()):
                sys.exit("Wrong value in " + c + " prediction column.")

        data = pd.merge(ground_truth, predicted, on="ID")

        if (len(ground_truth) != len(data)):
            sys.exit('Invalid tweet IDs in prediction.')

        # Compute Performance Measures HS
        acc_hs = accuracy_score(data["HS_x"], data["HS_y"])
        p_hs, r_hs, f1_hs, support = precision_recall_fscore_support(data["HS_x"], data["HS_y"], average = "macro")

        return acc_hs, p_hs, r_hs, f1_hs

    def evaluate_b(self,pred,gold):
        levels = ["HS", "TargetRange", "Aggressiveness"]

        ground_truth = pd.read_csv(gold, sep="\t", names=["ID", "Tweet-text", "HS", "TargetRange", "Aggressiveness"],
                                converters={0: str, 1: str, 2: int, 3: int, 4: int}, header=None, skiprows=1)

        predicted = pd.read_csv(pred, sep="\t", names=["ID"] + levels,
                                converters={0: str, 1: int, 2: int, 3: int}, header=None, skiprows=1)

        # Check length files
        if (len(ground_truth) != len(predicted)):
            sys.exit('Prediction and gold data have different number of lines.')

        # Check predicted classes
        for c in levels:
            gt_class = list(ground_truth[c].value_counts().keys())
            if not (predicted[c].isin(gt_class).all()):
                sys.exit("Wrong value in " + c + " prediction column.")

        data = pd.merge(ground_truth, predicted, on="ID")

        if (len(ground_truth) != len(data)):
            sys.exit('Invalid tweet IDs in prediction.')

        # Compute Performance Measures
        acc_levels = dict.fromkeys(levels)
        p_levels = dict.fromkeys(levels)
        r_levels = dict.fromkeys(levels)
        f1_levels = dict.fromkeys(levels)
        for l in levels:
            acc_levels[l] = accuracy_score(data[l + "_x"], data[l + "_y"])
            p_levels[l], r_levels[l], f1_levels[l], _ = precision_recall_fscore_support(data[l + "_x"], data[l + "_y"], average="macro")
        macro_f1 = np.mean(list(f1_levels.values()))

        # Compute Exact Match Ratio
        check_emr = np.ones(len(data), dtype=bool)
        for l in levels:
            check_label = data[l + "_x"] == data[l + "_y"]
            check_emr = check_emr & check_label
        emr = sum(check_emr) / len(data)

        return macro_f1, emr, acc_levels, p_levels, r_levels, f1_levels

    def main(self):
        # https://github.com/Tivix/competition-examples/blob/master/compute_pi/program/evaluate.py
        # as per the metadata file, input and output directories are the arguments


        # unzipped submission data is always in the 'res' subdirectory
        # https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions


        ref_dir = os.path.join(self.input_dir, 'ref')
        gold_standard = os.path.join(ref_dir, os.listdir(ref_dir)[0])
        lang = gold_standard.split('/')[-1].replace('.tsv', '')
        res_dir = os.path.join(self.input_dir, 'res')
        submission_path = os.path.join(res_dir, os.listdir(res_dir)[0])
        task = submission_path.split('/')[-1].replace('.tsv', '').split('_')[1]

        output_file = open(os.path.join(self.output_dir, 'scores.txt'), "w")
        if task == 'a':
            acc_hs, p_hs, r_hs, f1_hs = self.evaluate_a(submission_path, gold_standard)

            # the scores for the leaderboard must be in a file named "scores.txt"
            # https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions

            output_file.write("taskA_fscore: {0}\n".format(f1_hs))
            output_file.write("taskA_precision: {0}\n".format(p_hs))
            output_file.write("taskA_recall: {0}\n".format(r_hs))
            output_file.write("taskA_accuracy: {0}\n".format(acc_hs))
            print("taskA_fscore: {0}".format(f1_hs))
            print("taskA_precision: {0}".format(p_hs))
            print("taskA_recall: {0}".format(r_hs))
            print("taskA_accuracy: {0}".format(acc_hs))
        elif task == 'b':
            output_file.write("\n\nBeginning to evaluate task b:")
            macro_f1, emr, acc_levels, p_levels, r_levels, f1_levels = self.evaluate_b(submission_path, gold_standard)

            # the scores for the leaderboard must be in a file named "scores.txt"
            # https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions

            output_file.write("taskB_fscore_macro: {0}\n".format(macro_f1))
            output_file.write("taskB_emr: {0}\n".format(emr))
            output_file.write("taskB_fscore_HS: {0}\n".format(f1_levels["HS"]))
            output_file.write("taskB_precision_HS: {0}\n".format(p_levels["HS"]))
            output_file.write("taskB_recall_HS: {0}\n".format(r_levels["HS"]))
            output_file.write("taskB_accuracy_HS: {0}\n".format(acc_levels["HS"]))
            output_file.write("taskB_fscore_TR: {0}\n".format(f1_levels["TargetRange"]))
            output_file.write("taskB_precision_TR: {0}\n".format(p_levels["TargetRange"]))
            output_file.write("taskB_recall_TR: {0}\n".format(r_levels["TargetRange"]))
            output_file.write("taskB_accuracy_TR: {0}\n".format(acc_levels["TargetRange"]))
            output_file.write("taskB_fscore_AG: {0}\n".format(f1_levels["Aggressiveness"]))
            output_file.write("taskB_precision_AG: {0}\n".format(p_levels["Aggressiveness"]))
            output_file.write("taskB_recall_AG: {0}\n".format(r_levels["Aggressiveness"]))
            output_file.write("taskB_accuracy_AG: {0}\n".format(acc_levels["Aggressiveness"]))

            print("taskB_fscore_macro: {0}".format(macro_f1))
            print("taskB_emr: {0}n".format(emr))
            print("taskB_fscore_HS: {0}".format(f1_levels["HS"]))
            print("taskB_precision_HS: {0}".format(p_levels["HS"]))
            print("taskB_recall_HS: {0}".format(r_levels["HS"]))
            print("taskB_accuracy_HS: {0}".format(acc_levels["HS"]))
            print("taskB_fscore_TR: {0}".format(f1_levels["TargetRange"]))
            print("taskB_precision_TR: {0}".format(p_levels["TargetRange"]))
            print("taskB_recall_TR: {0}".format(r_levels["TargetRange"]))
            print("taskB_accuracy_TR: {0}".format(acc_levels["TargetRange"]))
            print("taskB_fscore_AG: {0}".format(f1_levels["Aggressiveness"]))
            print("taskB_precision_AG: {0}".format(p_levels["Aggressiveness"]))
            print("taskB_recall_AG: {0}".format(r_levels["Aggressiveness"]))
            print("taskB_accuracy_AG: {0}".format(acc_levels["Aggressiveness"]))

        output_file.close()
