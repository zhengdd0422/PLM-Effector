"""
@author: Zheng
run PLM-Effector
"""
import argparse
import subprocess
import os
import time

def main():
    parser = argparse.ArgumentParser(description="Run feature extraction and ensemble prediction")
    parser.add_argument("--usefile_id", type=str, required=True, help="Input file id")
    parser.add_argument("--effector_type", type=str, required=True, help="Effector type, e.g., T1SE, T4SE")
    parser.add_argument("--cuda", type=str, default="7", help="CUDA_VISIBLE_DEVICES id (default: 7)")
    args = parser.parse_args()

    # 记录开始时间
    start_time = time.time()

    # 路径
    py_exec = "/home/zhengdd/Softwares/anaconda2/envs/py39_cuda11.3/bin/python"

    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda

    # 根据 effector_type 选择模型
    if args.effector_type == "T4SE":
        pretrained_types = ["esm1", "esm2_t33", "ProtBert", "ProtT5"]
    else:
        pretrained_types = ["esm1", "esm2_t33", "ProtT5"]

    # Step1: 生成特征
    for model in pretrained_types:
        cmd = [
            py_exec,
            "transformers_pretrainedmodel-features_extract_4predict.py",
            "--pretrained_type", model,
            "--usefile_id", args.usefile_id,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)

    # Step2: 预测结果
    cmd = [
        py_exec,
        "ensemble_step12_4predict.py",
        "--usefile_id", args.usefile_id,
        "--effector_type", args.effector_type
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

    # 记录结束时间
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n Total runtime: {elapsed:.2f} seconds (~{elapsed/60:.2f} minutes)\n")
    with open("/home/zhengdd/bin/PLM-Effector_online/predict_out/" + args.usefile_id + ".finish.txt", "w") as file:
        pass

if __name__ == "__main__":
    main()

