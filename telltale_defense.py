import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from autoencoder import train_autoencoder
import os
def norm(dataset):
    mean = np.mean(dataset, axis=0)
    std = np.std(dataset, axis=0)
    normalize = (dataset - mean) / std
    return normalize

def dim_transp(trajectory_loss_poison_re, trajectory_loss_clean_re, trajectory_loss_re):
    poison_trace = []
    clean_trace = []
    all_trace = []
    for i in range(np.asmatrix(trajectory_loss_poison_re).shape[1]):
        f = []
        for j in range(np.asmatrix(trajectory_loss_poison_re).shape[0]):
            f.append(trajectory_loss_poison_re[j][i])
        poison_trace.append(f)
    for i in range(np.asmatrix(trajectory_loss_clean_re).shape[1]):
        f = []
        for j in range(np.asmatrix(trajectory_loss_clean_re).shape[0]):
            f.append(trajectory_loss_clean_re[j][i])
        clean_trace.append(f)
    for i in range(np.asmatrix(trajectory_loss_re).shape[1]):
        f = []
        for j in range(np.asmatrix(trajectory_loss_re).shape[0]):
            f.append(trajectory_loss_re[j][i])
        all_trace.append(f)
    print(np.asmatrix(poison_trace).shape)
    print(np.asmatrix(clean_trace).shape)
    print(np.asmatrix(all_trace).shape)
    return poison_trace, clean_trace, all_trace

def display(loss_trace):
    label_name = ["benign_loss", "poison_loss"]
    plt.figure(figsize=(10, 7), dpi=300)
    for k in range(2):
        avg = np.mean(loss_trace[k], axis=1)
        std = np.std(loss_trace[k], axis=1)
        r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))
        r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))
        plt.plot(np.arange(len(avg)), avg, label=label_name[k])
        plt.fill_between(np.arange(len(avg)), r1, r2, alpha=0.2)

def curve_plot(clean_trace, poison_trace):
    loss_trace = [clean_trace, poison_trace]
    display(loss_trace)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.legend(loc='upper right', fontsize=18)
    plt.savefig('./figure/loss_curve.png')
    print("  The loss curve has been saved at './figure/loss_curve.png'.")

def dim_reduction(s, d, net, trajectory_loss_re, trajectory_loss_clean_re):
    net.load_state_dict(torch.load('./LSTM classifiers/Autoencoder.pth'))
    net.eval()
    pred_clean, _ = net(trajectory_loss_re[:len(trajectory_loss_clean_re)].view(-1, 1, s))
    pred_clean = np.array(pred_clean.squeeze(1).detach().numpy())
    pred_poison, _ = net(trajectory_loss_re[len(trajectory_loss_clean_re):].view(-1, 1, s))
    pred_poison = np.array(pred_poison.squeeze(1).detach().numpy())
    return pred_clean, pred_poison

def t_sne(pred_clean_fft, pred_poison_fft, pred_fft):
    print("  t-SNE visualization...")
    labels = np.concatenate(
        [np.ones(pred_clean_fft.shape[0], dtype=int), np.zeros(pred_poison_fft.shape[0], dtype=int)])
    tsne = TSNE(n_components=2, random_state=0, n_iter=1500)
    all_samples_tsne = tsne.fit_transform(pred_fft)
    colors = np.array(['red', 'cornflowerblue'])
    plt.figure(figsize=(20, 15), dpi=200)
    plt.scatter(all_samples_tsne[:, 0], all_samples_tsne[:, 1], c=colors[labels], cmap='coolwarm', alpha=0.5)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
    plt.savefig('./figure/t_sne.png', bbox_inches='tight', pad_inches=0.1)
    print("  The t-SNE visualization has been saved at './figure/t_sne.png'.")
    return all_samples_tsne

def clustering_dbscan(all_samples):
    print("  Clustering using DBSCAN...")
    cluster_data = all_samples
    eps, min_samples = 5, 5
    thecluster = DBSCAN(eps=eps, min_samples=min_samples).fit(cluster_data)
    labels = thecluster.labels_
    return labels

def main():
    trajectory_loss_re = np.load(file="./data_loss_trace/trajectory_loss_re.npy")
    trajectory_loss_clean_re = np.load(file="./data_loss_trace/trajectory_loss_clean_re.npy")
    trajectory_loss_poison_re = np.load(file="./data_loss_trace/trajectory_loss_poison_re.npy")
    print("The truncated loss trajectory data has been successfully loaded.")
    if not os.path.exists("./figure"):
        os.mkdir("./figure")
    poison_trace, clean_trace, all_trace = dim_transp(trajectory_loss_poison_re, trajectory_loss_clean_re, trajectory_loss_re)
    curve_plot(clean_trace, poison_trace)
    trajectory_loss_re = torch.from_numpy(np.asmatrix(trajectory_loss_re)).float()
    s, d, net = train_autoencoder(trajectory_loss_re)
    pred_clean, pred_poison = dim_reduction(s, d, net, trajectory_loss_re, trajectory_loss_clean_re)
    pred_clean, pred_poison = norm(np.array(pred_clean)), norm(np.array(pred_poison))
    pred_clean_fft, pred_poison_fft = np.abs(np.fft.fft(pred_clean)), np.abs(np.fft.fft(pred_poison))
    pred_fft = np.concatenate([pred_clean_fft, pred_poison_fft])
    all_samples = t_sne(pred_clean_fft, pred_poison_fft, pred_fft)
    labels = clustering_dbscan(all_samples)
    c1, c2 = 0, 0
    for i in range(len(pred_poison)):
        if labels[len(pred_clean) + i] == 0:
            c1 += 1
    for i in range(len(pred_clean)):
        if labels[i] == 0:
            c2 += 1

    print("\n***** Detection Accuracy & FPR *****")
    print("Det.Acc: {:.3f}% {}/{}".format((len(pred_poison) - c1) / len(pred_poison) * 100, len(pred_poison) - c1,
                                      len(pred_poison)))
    print("FPR: {:.3f}% {}/{}".format((len(pred_clean) - c2) / len(pred_clean) * 100, len(pred_clean) - c2,
                                      len(pred_clean)))

if __name__ == "__main__":
    main()
