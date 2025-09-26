# -*- coding: utf-8 -*-
"""
@author : Dandan Zheng, 2025/09/09
 The algorithm was implemented using Python 3.9, TensorFlow 2.9, env: py39_cuda11.3
    model1: esm1b Nter, MLP, CNN_attention, CNN_attention, CNN_attention, Attention for T1SE, T2SE, T3SE, T4SE, T6SE, respectively.
    model2: esm2_t33 Nter, MLP, Attention, MLP, Attention, CNN for T1SE, T2SE, T3SE, T4SE, T6SE, respectively.
    model3: protT5 Nter, CNN, CNN, CNN, Attention, CNN_attention for T1SE, T2SE, T3SE, T4SE, T6SE, respectively.
    model4: esm1b Cter, CNN_attention, CNN, Attention, CNN_attention, MLP for T1SE, T2SE, T3SE, T4SE, T6SE, respectively.
    model5: esm2_t33 Cter, CNN_attention,  CNN, Attention, CNN_attention, CNN_attention for T1SE, T2SE, T3SE, T4SE, T6SE, respectively.
    model6: protT5 Cter, CNN, CNN, Attention, MLP, CNN_attention for T1SE, T2SE, T3SE, T4SE, T6SE, respectively.
    model7: protBert Nter, CNN  for T4SE
    model8: protBert Cter, CNN_attention for T4SE
 """
import torch
import os
import gc
import numpy as np
from xgboost import XGBClassifier
from argparse import ArgumentParser
from trainer import loadmodel_4predict, loadmodel_4test
from utils import set_seed, pool_features, load_predict_numpy_nopool
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(__file__)
data_path = os.path.join(BASE_DIR, "tmp")
model_path = os.path.join(BASE_DIR, "trained_models")
out_path = os.path.join(BASE_DIR, "predict_out")

def main(args):
    set_seed(42)
    ##############################
    # step 1
    ##############################
    if args.effector_type == "T4SE":
        test_OOF_dict = {f"model{i+1}": {"probs": []} for i in range(8)}
    else:
        test_OOF_dict = {f"model{i+1}": {"probs": []} for i in range(6)}

    for fold in range(5):
        print(f"Fold{fold}: Start to process model1\n")
        x_test1_embed, x_test1_atten, ids_test1 = load_predict_numpy_nopool(data_path, args.usefile_id + "_esm1_Nterminal.npz")
        if args.effector_type == "T1SE":
            x_test1 = pool_features(x_test1_embed, x_test1_atten, pooling="mean").numpy()
        else:
            x_test1 = x_test1_embed
        x_test1 = torch.from_numpy(x_test1).float()
     
        test_preds1, test_probs1 = loadmodel_4test(model_path, f"{args.effector_type}_model1_fold{fold}.pth", x_test1, device)
        test_OOF_dict["model1"]["probs"].append(test_probs1)
        del x_test1_embed, x_test1_atten, x_test1, test_probs1
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Fold{fold}:Start to process model2\n")
        x_test2_embed, x_test2_atten, ids_test2 = load_predict_numpy_nopool(data_path, args.usefile_id + "_esm2_t33_Nterminal.npz")
        if args.effector_type == "T1SE" or args.effector_type == "T3SE":
            x_test2 = pool_features(x_test2_embed, x_test2_atten, pooling="mean").numpy()
        else:
            x_test2 = x_test2_embed
        x_test2 = torch.from_numpy(x_test2).float()
        with torch.no_grad():
            test_probs2 = loadmodel_4predict(model_path, f"{args.effector_type}_model2_fold{fold}.pth", x_test2, device)
        test_OOF_dict["model2"]["probs"].append(test_probs2)
        del x_test2_embed, x_test2_atten, x_test2, test_probs2
        torch.cuda.empty_cache()
        gc.collect()

        # model3
        print(f"Fold{fold}:Start to process model3\n")
        x_test3_embed, x_test3_atten, ids_test3 = load_predict_numpy_nopool(data_path, args.usefile_id + "_ProtT5_Nterminal.npz")
        x_test3 = torch.from_numpy(x_test3_embed).float()
        with torch.no_grad():
            test_probs3 = loadmodel_4predict(model_path, f"{args.effector_type}_model3_fold{fold}.pth", x_test3, device)
        test_OOF_dict["model3"]["probs"].append(test_probs3)
        del x_test3_embed, x_test3_atten, x_test3, test_probs3
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Fold{fold}:Start to process model4\n")
        x_test4_embed, x_test4_atten, ids_test4 = load_predict_numpy_nopool(data_path,  args.usefile_id + "_esm1_Cterminal.npz")
        if args.effector_type == "T6SE":
            x_test4 = pool_features(x_test4_embed, x_test4_atten, pooling="mean").numpy()
        else:
            x_test4 = x_test4_embed
        x_test4 = torch.from_numpy(x_test4).float()
        with torch.no_grad():
            test_probs4 = loadmodel_4predict(model_path, f"{args.effector_type}_model4_fold{fold}.pth", x_test4, device)
        test_OOF_dict["model4"]["probs"].append(test_probs4)
        del x_test4_embed, x_test4_atten, x_test4, test_probs4
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Fold{fold}:Start to process model5\n")
        x_test5_embed, x_test5_atten, ids_test5 = load_predict_numpy_nopool(data_path, args.usefile_id + "_esm2_t33_Cterminal.npz")
        x_test5 = torch.from_numpy(x_test5_embed).float()
        with torch.no_grad():
            test_probs5 = loadmodel_4predict(model_path, f"{args.effector_type}_model5_fold{fold}.pth", x_test5, device)
        test_OOF_dict["model5"]["probs"].append(test_probs5)
        del x_test5_embed, x_test5_atten, x_test5, test_probs5
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Fold{fold}:Start to process model6\n")
        x_test6_embed, x_test6_atten, ids_test6 = load_predict_numpy_nopool(data_path,  args.usefile_id + "_ProtT5_Cterminal.npz")
        if args.effector_type == "T4SE":
            x_test6 = pool_features(x_test6_embed, x_test6_atten, pooling="mean").numpy()
        else:
            x_test6 = x_test6_embed
        x_test6 = torch.from_numpy(x_test6).float()
        with torch.no_grad():
            test_probs6 = loadmodel_4predict(model_path, f"{args.effector_type}_model6_fold{fold}.pth", x_test6, device)
        test_OOF_dict["model6"]["probs"].append(test_probs6)
        del x_test6_embed, x_test6_atten, x_test6, test_probs6
        torch.cuda.empty_cache()
        gc.collect()

        if args.effector_type == "T4SE":
            print(f"Fold{fold}:Start to process model7\n")
            x_test7_embed, x_test7_atten, ids_test7 = load_predict_numpy_nopool(data_path, args.usefile_id + "_ProtBert_Nterminal.npz")
            x_test7 = torch.from_numpy(x_test7_embed).float()
            with torch.no_grad():
                test_probs7 = loadmodel_4predict(model_path, f"{args.effector_type}_model7_fold{fold}.pth", x_test7, device)
            test_OOF_dict["model7"]["probs"].append(test_probs7)
            del x_test7_embed, x_test7_atten, x_test7, test_probs7
            torch.cuda.empty_cache()
            gc.collect()

            print(f"Fold{fold}:Start to process model8\n")
            x_test8_embed, x_test8_atten, ids_test8 = load_predict_numpy_nopool(data_path, args.usefile_id + "_ProtBert_Cterminal.npz")
            x_test8 = torch.from_numpy(x_test8_embed).float()
            with torch.no_grad():
                test_probs8 = loadmodel_4predict(model_path, f"{args.effector_type}_model8_fold{fold}.pth", x_test8, device)
            test_OOF_dict["model8"]["probs"].append(test_probs8)
            del x_test8_embed, x_test8_atten, x_test8, test_probs8
            torch.cuda.empty_cache()
            gc.collect()

    # check whether seq_id is equal
    all_ids = [ids_test1, ids_test2, ids_test3, ids_test4, ids_test5, ids_test6]
    id_names = ["ids_test1", "ids_test2", "ids_test3", "ids_test4", "ids_test5", "ids_test6"]
    if args.effector_type == "T4SE":
        all_ids.extend([ids_test7, ids_test8])
        id_names.extend(["ids_test7", "ids_test8"])
    for i in range(1, len(all_ids)):
        if not np.array_equal(all_ids[0], all_ids[i]):
            raise ValueError(f"Error：{id_names[0]} != {id_names[i]}\n"
                             f"{id_names[0]} length: {len(all_ids[0])}, content: {all_ids[0][:5]}...\n"
                             f"{id_names[i]} length: {len(all_ids[i])}, content: {all_ids[i][:5]}...")

    test_probs_list = []
    if args.effector_type == "T4SE":
        model_number = 8
    else:
        model_number = 6
    for i in range(model_number):
        model_name = f"model{i+1}"
        test_probs_avg = np.mean(np.stack(test_OOF_dict[model_name]["probs"], axis=0), axis=0)
        test_probs_list.append(test_probs_avg.reshape(-1, 1))

    x_test_stacking = np.hstack(test_probs_list)

    ##############################
    # step2
    ##############################
    best_threshold = {
        "T1SE": 0.5,
        "T2SE": 0.7,
        "T3SE": 0.7,
        "T4SE": 0.6,
        "T6SE": 0.5
    }
    meta_model = XGBClassifier()
    best_model_path = os.path.join(args.data_path,"trained_models", f"{args.effector_type}_XGB_stackingmeta_model.json")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"XGBoost is not exist: {best_model_path}\n")
    meta_model.load_model(best_model_path)
    final_probs = meta_model.predict_proba(x_test_stacking)[:, 1]
    out_file = os.path.join(out_path, f"{args.usefile_id}_PLM-Effector_out.txt")
    with open(out_file, "w") as f:
        if args.effector_type == "T4SE":
            header = "seq_id\tmodel1\tmodel2\tmodel3\tmodel4\tmodel5\tmodel6\tmodel7\tmodel8\tstacking\n"
        else:
            header = "seq_id\tmodel1\tmodel2\tmodel3\tmodel4\tmodel5\tmodel6\tstacking\n"
        f.write(header)
        for i, sid in enumerate(ids_test1):
            model_probs = x_test_stacking[i, :]
            # if final_probs[i] >= best_threshold[args.effector_type] and all(p >= 0.5 for p in model_probs):
            if final_probs[i] >= best_threshold[args.effector_type]:
                if args.effector_type == "T4SE":
                    line = [sid] + [f"{x_test_stacking[i,j]:.4f}" for j in range(8)] + [f"{final_probs[i]:.4f}"] + ["1"]
                else:
                    line = [sid] + [f"{x_test_stacking[i,j]:.4f}" for j in range(6)] + [f"{final_probs[i]:.4f}"] + ["1"]
                f.write("\t".join(line) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--effector_type", type=str, default="T2SE", choices=['T1SE', 'T2SE', 'T3SE', 'T4SE', 'T6SE'])
    parser.add_argument("--usefile_id", type=str, default='1', help="the path of the data sets")
    args = parser.parse_args()
    main(args)

