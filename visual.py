import json
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curve_from_json():
    json_path1 = "ckpts/qwen2vl_7b_one_num/checkpoint-1410/trainer_state.json"

    with open(json_path1, "r") as f:
        log_data = json.load(f)

    steps, losses = [], []
    for entry in log_data["log_history"]:
        steps.append(entry["step"])
        losses.append(entry["loss"])


    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses)
    plt.xlabel("Training Steps")
    plt.ylabel("Training Loss")
    plt.grid(True)
    plt.savefig('./imgs/qwen2vl_7b_3ep_var_trainloss.png', bbox_inches="tight")

    # steps, grad_norm = [], []
    # for entry in log_data["log_history"]:
    #     steps.append(entry["step"])
    #     grad_norm.append(entry["grad_norm"])

    # plt.figure(figsize=(8, 5))
    # plt.plot(steps, grad_norm)
    # plt.xlabel("Training Steps")
    # plt.ylabel("Training Grad Norm")
    # plt.grid(True)
    # plt.savefig('./imgs/ms_gradnorm.png', bbox_inches="tight")

def plot_loss_curve_from_log(path):
    with open(path, "r") as f:
        log_data = f.readlines()

    losses = []
    # accuracies = []
    for i in range(0,len(log_data),2):
        chunk = log_data[i:i+2]            # 取第 i 和 i+1 行
        loss = [float(line.split('\t')[2]) for line in chunk]
        avg  = sum(loss) / len(loss)

        losses.append(avg)
        # accuracies.append(float(acc))

    plt.figure(figsize=(16, 9))
    plt.plot(losses, label='Loss', color='orange')
    # plt.plot(accuracies, label='Accuracy', color='blue')
    # plt.xticks(range(0, len(losses), len(losses) // 8),  np.linspace(1, 8, 8, dtype=int))
    plt.xlabel("Training Steps")
    plt.ylabel("Value")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig('./imgs/youcook_projloss.png', bbox_inches="tight")


if __name__ == "__main__":
    plot_loss_curve_from_json()
    # plot_loss_curve_from_log("ckpts/qwen2vl_7b_proj_youcook2/losses.log")
