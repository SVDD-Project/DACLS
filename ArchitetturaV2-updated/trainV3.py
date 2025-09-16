import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import savemat

import torch
from torchvision.utils import save_image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

from dcgan_modelV2 import RefinerGenerator
from utilsV3 import pt_load_dataset

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert max_size > 0
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data:
            element = torch.unsqueeze(element.data, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    tmp = self.data[idx].clone()
                    self.data[idx] = element
                    to_return.append(tmp)
                else:
                    to_return.append(element)
        return torch.cat(to_return)

def compute_discriminator_loss(netD, split, criterion, device, batch_size=32):
    """
    Calcola la loss media del discriminatore su uno split specificato.
    
    Args:
        netD: modello discriminatore
        split: 'Validation' o 'Test' (o 'Training')
        criterion: funzione di loss (es. BCEWithLogitsLoss)
        device: dispositivo ('cuda' o 'cpu')
        batch_size: batch size per il DataLoader
    """
    # Carica i dataloader di bonafide e spoof
    bonafide_loss_dload = pt_load_dataset(batch_size=batch_size, split=split, lbl='bonafide')
    spoof_loss_dload = pt_load_dataset(batch_size=batch_size, split=split, lbl='spoof')

    netD.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for (imgs_b, _), (imgs_s, _) in zip(bonafide_loss_dload, spoof_loss_dload):
            imgs_b = imgs_b.to(device)
            imgs_s = imgs_s.to(device)

            # Bonafide = 1, Spoof = 0
            labels_b = torch.ones(imgs_b.size(0), device=device)
            labels_s = torch.zeros(imgs_s.size(0), device=device)

            # Forward + loss
            outputs_b = netD(imgs_b).view(-1)
            outputs_s = netD(imgs_s).view(-1)

            #print("output_b", outputs_b)
            #print("output_s", outputs_s)

            loss_b = criterion(outputs_b, labels_b)
            loss_s = criterion(outputs_s, labels_s)

            total_loss += (loss_b.item() + loss_s.item()) / 2  # media dei due batch
            num_batches += 1  # una iterazione dello zip = un batch medio


    netD.train()
    return total_loss / num_batches


def evaluate_discriminator(netD, split, device, batch_size = 32):
    """
    Valuta il discriminatore su un dataset specificato dallo split.
    
    Args:
        netD: modello discriminatore
        split: 'Validation' o 'Test' (o 'Training' se vuoi)
        batch_size: batch size per il DataLoader
        device: dispositivo ('cuda' o 'cpu')
    """
    # Carica i dataloader di bonafide e spoof
    bonafide_gen_dload = pt_load_dataset(batch_size=batch_size, split=split, lbl='bonafide')
    spoof_gen_dload = pt_load_dataset(batch_size=batch_size, split=split, lbl='spoof')

    y_true = []
    y_scores = []

    netD.eval()
    # Itera separatamente sui dataloader
    with torch.no_grad():
        # Bonafide
        for imgs_b, _ in bonafide_gen_dload:
            imgs_b = imgs_b.to(device)
            outputs_b = netD(imgs_b).view(-1)
            y_scores.extend(outputs_b.cpu().numpy())
            y_true.extend([1] * imgs_b.size(0))

        # Spoof
        for imgs_s, _ in spoof_gen_dload:
            imgs_s = imgs_s.to(device)
            outputs_s = netD(imgs_s).view(-1)
            y_scores.extend(outputs_s.cpu().numpy())
            y_true.extend([0] * imgs_s.size(0))

    # Predizioni usando soglia 0.5
    y_pred = [1 if s >= 0.5 else 0 for s in y_scores]

    # Metriche
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

    print(f"Evaluation results on {split} set:")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
          f"F1: {f1:.4f}, AUC: {auc:.4f}, EER: {eer:.4f}")

    netD.train()
    return accuracy, precision, recall, f1, auc, eer



def save_all_codes(run_path="runV3.py", train_path="trainV3.py", dcgan_path="dcgan_modelV2.py", output_file="all_codes.txt"):
    

    paths = [run_path, train_path, dcgan_path]
    with open(output_file, "w") as outfile:
        for path in paths:
            outfile.write(f"### File: {path}\n\n")
            if os.path.exists(path):
                with open(path, "r") as infile:
                    outfile.write(infile.read())
            else:
                outfile.write(f"File {path} non trovato.\n")
            outfile.write("\n\n" + "#"*80 + "\n\n")

def train_dcgan(device,
                nz,
                lr_g,
                lr_d,
                beta1,
                netD,
                netG,
                spoof_dload,
                bonafide_dload,
                num_epochs,
                result_path_numbered,
                lambda_l1,
                lambda_mse,
                lambda_fm,
                update_d_every,
                dropout_p,
                noise_std):


    val_metrics = []
    test1_metrics = []
    test2_metrics = []

    torch.manual_seed(42)
    device = torch.device(device)

    result_path = result_path_numbered
    os.makedirs(result_path, exist_ok=True)

    img_path = os.path.join(result_path, "generated_images")
    os.makedirs(img_path, exist_ok=True)

    criterion = torch.nn.BCELoss()
    l1_loss = torch.nn.L1Loss()
    mse_loss = torch.nn.MSELoss()

    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999)) #weight_decay=weight_decay)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, 0.999))

    G_losses, D_losses = [], []
    precision_list, recall_list, f1_list, auc_list, eer_list, acc_list = [], [], [], [], [], []
    accG_list, lossG_list = [], []
    adv_loss_list, l1_loss_list, mse_loss_list, fm_loss_list = [], [], [], []
    val_metrics = []
    val_loss_d_list = []
    errD_real_list = []
    errD_fake_list = []

    early_stop_patience = 15
    best_val_loss = float("inf")
    epochs_no_improve = 0

    replay_buffer = ReplayBuffer(max_size=50)



    for epoch in range(num_epochs):

        y_true, y_scores = [], []
        total_loss_D = 0.0
        total_loss_G = 0.0
        total_l1_loss = 0.0
        total_mse_loss = 0.0
        total_fm_loss = 0.0
        correct_fake = 0.0
        total_fake = 0.0
        total_adv_loss = 0.0
        step_count = 0

        len_batch = min(len(bonafide_dload), len(spoof_dload))

        # ciclo principale: zip dei due dataloader
        for i, (real_data, fake_data) in enumerate(zip(bonafide_dload, spoof_dload), 0):
            step_count += 1

            # === Real e Fake dal dataset ===
            real_images, _ = real_data
            real_images = real_images.to(device)
            

            input_fake, _ = fake_data
            input_fake = input_fake.to(device)

            # Genera nuovi fake raffinati
            fake_images = netG(input_fake)
            fake_images_for_D = replay_buffer.push_and_pop(fake_images.detach())
            
            batch_size = real_images.size(0)
            
            # === Aggiorna Discriminatore ===
            optimizerD.zero_grad()

            # Etichette reali (label smoothing)
            label_real = torch.full((batch_size,), 0.9, device=device)
            label_fake = torch.full((batch_size,), 0.1, device=device)

            # Output D su immagini reali
            output_real = netD(real_images).view(-1)
            errD_real = criterion(output_real, label_real)

            # Output D su immagini fake
            output_fake = netD(fake_images_for_D).view(-1)
            errD_fake = criterion(output_fake, label_fake)

            # Loss totale e backward
            errD = errD_real + errD_fake
            errD.backward()

            # Applica aggiornamento *ogni N step*
            if step_count % update_d_every == 0:
                optimizerD.step()

            D_x = output_real.mean().item()
            D_G_z1 = output_fake.mean().item()

            errD_real_list.append(errD_real.item())
            errD_fake_list.append(errD_fake.item())

            # --- Generatore ---
            netG.zero_grad()

            output_gen = netD(fake_images).view(-1)
            label_gen = torch.full((output_gen.size(0),), 1.0, device=device)
            adv_loss = criterion(output_gen, label_gen)

            # --- Matching dimensioni ---
            min_batch_size = min(fake_images.size(0), real_images.size(0))
            fake_images = fake_images[:min_batch_size]
            real_images = real_images[:min_batch_size]

            l1 = l1_loss(fake_images, real_images)
            mse = mse_loss(fake_images, real_images)

            # Feature matching: real senza grad, fake con grad
            with torch.no_grad():
                _, real_feats = netD(real_images, return_features=True)
            _, fake_feats = netD(fake_images, return_features=True)

            fm_loss = mse_loss(fake_feats, real_feats)

            errG = adv_loss + lambda_l1 * l1 + lambda_mse * mse + lambda_fm * fm_loss
            errG.backward()
            D_G_z2 = output_gen.mean().item()
            optimizerG.step()

            # --- Logging ---
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            total_loss_D += errD.item()
            total_loss_G += errG.item()
            total_adv_loss += adv_loss.item()
            total_l1_loss += l1.item()
            total_mse_loss += mse.item()
            total_fm_loss += fm_loss.item()
            correct_fake += (output_gen > 0.5).sum().item()
            total_fake += batch_size

            y_true.extend([1]*batch_size + [0]*batch_size)
            y_scores.extend(output_real.detach().cpu().numpy().tolist() +
                            output_fake.detach().cpu().numpy().tolist())

            if i % 50 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] "
                    f"[Batch {i}/{len_batch}] "
                    f"Loss_D: {errD.item():.4f} "
                    f"Loss_G: {errG.item():.4f} "
                    f"D(x): {D_x:.4f} "
                    f"D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")


        y_pred = [1 if s >= 0.5 else 0 for s in y_scores]

        #queste 4 righe sotto servono per far avere la stessa lunghezza a y_true e y_pred
        #(se subset_size e il numero di immagini fake sono diversi)
        min_len = min(len(y_true), len(y_pred), len(y_scores))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        y_scores = y_scores[:min_len]
        assert len(y_true) == len(y_pred), f"Inconsistent lengths: {len(y_true)} vs {len(y_pred)}"

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_scores)
        accuracy = accuracy_score(y_true, y_pred)

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

        avg_loss_G = total_loss_G / len_batch
        accuracy_G = correct_fake / total_fake
        avg_adv_loss = total_adv_loss / len_batch
        avg_l1_loss = total_l1_loss / len_batch
        avg_mse_loss = total_mse_loss / len_batch
        avg_fm_loss = total_fm_loss / len_batch

        print(f"--- Epoch {epoch+1} metrics ---")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, "
              f"AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, EER: {eer:.4f}")
        print(f"Generator Accuracy: {accuracy_G:.4f}, Generator Loss: {avg_loss_G:.4f}")
        print(f"L1 Loss: {avg_l1_loss:.4f}, MSE Loss: {avg_mse_loss:.4f}, FM Loss: {avg_fm_loss:.4f}")

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        auc_list.append(auc)
        acc_list.append(accuracy)
        eer_list.append(eer)
        accG_list.append(accuracy_G)
        lossG_list.append(avg_loss_G)
        adv_loss_list.append(avg_adv_loss)
        l1_loss_list.append(avg_l1_loss)
        mse_loss_list.append(avg_mse_loss)
        fm_loss_list.append(avg_fm_loss)

        with torch.no_grad():
            example_input = next(iter(spoof_dload))[0].to(device)  # Prendi una batch dal dataset fake
            refined_fake = netG(example_input).detach().cpu()

            # Salva immagine per l'epoca corrente
            save_image(refined_fake, f"{img_path}/refined_epoch_{epoch + 1}.png", normalize=True)

            # Salva immagine finale e formati aggiuntivi
            save_image(refined_fake, f"{img_path}/final_result.png", normalize=True)
            npy_array = refined_fake.numpy()
            np.save(f"{img_path}/final_result.npy", npy_array)
            mat_array = np.squeeze(npy_array, axis=1)
            savemat(f"{img_path}/final_result.mat", {"spectrograms": mat_array})

            # === VALIDAZIONE INTERMEDIA ===
            print(f"### VALIDATION EPOCH {epoch + 1} ###")
            val = evaluate_discriminator(netD, 'Validation', device)
            val_metrics.append(val)

            # Salvo la validation loss
            val_loss_d = compute_discriminator_loss(netD, 'Validation', criterion, device)

            #Criteri early stopping
            if val_loss_d < best_val_loss:
                best_val_loss = val_loss_d
                epochs_no_improve = 0
                torch.save(netD.state_dict(), os.path.join(result_path, "best_netD.pth"))
            else:
                epochs_no_improve += 1

            # L'early stopping parte solo dopo 50 epoche
            if epoch + 1 >= 30 and epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        val_loss_d_list.append(val_loss_d)

    def plot_loss_curves(G_losses, D_losses, result_path):
        plt.figure(figsize=(10, 5))
        plt.plot(G_losses, label="Generator Loss")
        plt.plot(D_losses, label="Discriminator Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curves")
        plt.grid(True)
        plt.savefig(f"{result_path}/loss_curves.png")
        plt.close()

    def plot_training_metrics(epochs, precision, recall, f1, auc, eer, result_path):
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, precision, label="Precision")
        plt.plot(epochs, recall, label="Recall")
        plt.plot(epochs, f1, label="F1 Score")
        plt.plot(epochs, auc, label="AUC")
        plt.plot(epochs, eer, label="EER")
        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.title("Training Metrics over Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{result_path}/training_metrics.png")
        plt.close()

    def plot_discriminator_metrics(epochs, acc_D, D_losses, len_batch, result_path, errD_real_list, errD_fake_list):
        min_len = min(len(epochs), len(acc_D), len(D_losses) // len_batch)

        avg_loss_D = [
            np.mean(D_losses[i * len_batch:(i + 1) * len_batch])
            for i in range(min_len)
        ]

        avg_loss_real = [
            np.mean(errD_real_list[i * len_batch:(i + 1) * len_batch])
            for i in range(min_len)
        ]

        avg_loss_fake = [
            np.mean(errD_fake_list[i * len_batch:(i + 1) * len_batch])
            for i in range(min_len)
        ]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs[:min_len], acc_D[:min_len], label="Discriminator Accuracy")
        plt.plot(epochs[:min_len], avg_loss_D, label="Total Loss D")
        plt.plot(epochs[:min_len], avg_loss_real, label="Loss on Real")
        plt.plot(epochs[:min_len], avg_loss_fake, label="Loss on Fake")
        plt.xlabel("Epoch")
        plt.ylabel("Loss / Accuracy")
        plt.title("Discriminator Metrics per Epoch")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{result_path}/discriminator_metrics.png")
        plt.close()

        return avg_loss_D, avg_loss_real, avg_loss_fake


    def plot_generator_metrics(epochs, acc_G, loss_G, result_path):
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, acc_G, label="Generator Accuracy")
        plt.plot(epochs, loss_G, label="Generator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Generator Accuracy and Loss per Epoch")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{result_path}/generator_metrics.png")
        plt.close()

    def save_metrics_table(epochs, metrics_dict, result_path):
        # Calcola la lunghezza minima tra epochs e tutte le metriche
        min_len = min([len(epochs)] + [len(v) for v in metrics_dict.values()])

        # Crea il DataFrame con lunghezze coerenti
        df = pd.DataFrame({"Epoch": epochs[:min_len]})
        for key, value in metrics_dict.items():
            df[key] = value[:min_len]

        # Salva in CSV e TXT
        df.to_csv(f"{result_path}/metrics_summary.csv", index=False)
        df.to_string(buf=open(f"{result_path}/metrics_summary.txt", "w"), index=False)


    def save_final_evaluation(val, test1, test2, result_path):
        df = pd.DataFrame({
            "Set": ["Validation", "Test T01", "Test T02"],
            "Accuracy": [val[0], test1[0], test2[0]],
            "Precision": [val[1], test1[1], test2[1]],
            "Recall": [val[2], test1[2], test2[2]],
            "F1": [val[3], test1[3], test2[3]],
            "AUC": [val[4], test1[4], test2[4]],
            "EER": [val[5], test1[5], test2[5]],
        })
        with open(f"{result_path}/final_evaluation_metrics.txt", "w") as f:
            f.write(df.to_string(index=False))

    # === Chiamate di funzione organizzate ===

    # Epochs e lunghezza minima
    num_recorded_epochs = len(precision_list)
    epochs = list(range(1, num_recorded_epochs + 1))
    min_len = min(
        len(epochs), len(precision_list), len(recall_list), len(f1_list), len(auc_list),
        len(acc_list), len(val_metrics), len(eer_list), len(accG_list), len(lossG_list),
        len(D_losses) // len_batch, len(val_loss_d_list),
        len(adv_loss_list), len(l1_loss_list), len(mse_loss_list), len(fm_loss_list)
    )

    # Plot
    plot_loss_curves(G_losses, D_losses, result_path)
    plot_training_metrics(epochs[:min_len], precision_list[:min_len], recall_list[:min_len],
                        f1_list[:min_len], auc_list[:min_len], eer_list[:min_len], result_path)
    avg_loss_D, avg_loss_real, avg_loss_fake = plot_discriminator_metrics(
    epochs, acc_list, D_losses, len_batch, result_path, errD_real_list, errD_fake_list
    )

    plot_generator_metrics(epochs[:min_len], accG_list[:min_len], lossG_list[:min_len], result_path)

    # Salva tabella riassuntiva per epoche
    metrics_dict = {
        "Precision": precision_list[:min_len],
        "Recall": recall_list[:min_len],
        "F1": f1_list[:min_len],
        "AUC": auc_list[:min_len],
        "Accuracy_D": acc_list[:min_len],
        "val_accuracy_D": [val[0] for val in val_metrics[:min_len]],
        "EER": eer_list[:min_len],
        "Accuracy_G": accG_list[:min_len],
        "Loss_G": lossG_list[:min_len],
        "Loss_D": avg_loss_D,
        "Val_Loss_D": val_loss_d_list[:min_len],
        "Adv_loss": adv_loss_list[:min_len],
        "L1_Loss": l1_loss_list[:min_len],
        "MSE_Loss": mse_loss_list[:min_len],
        "FM_Loss": fm_loss_list[:min_len],
        "Loss_D_Real": avg_loss_real,
        "Loss_D_Fake": avg_loss_fake,

    }
    save_metrics_table(epochs[:min_len], metrics_dict, result_path)

    # Salva codici usati
    save_all_codes()
    print("File 'all_codes.txt' creato con il contenuto di run.py, train.py e dcgan_model.py")

    # Carico il modello migliore salvato
    netD.load_state_dict(torch.load(os.path.join(result_path, "best_netD.pth")))

    # === VALIDATION & TEST ===
    print("### VALIDATION ###")
    val = evaluate_discriminator(netD, 'Validation', device)
    val_metrics.append(val)

    print("### TEST T01 ###")
    test1 = evaluate_discriminator(netD, 'T01', device)
    test1_metrics.append(test1)

    print("### TEST T02 ###")
    test2 = evaluate_discriminator(netD, 'T02', device)
    test2_metrics.append(test2)

    # Salvataggio finale
    save_final_evaluation(val, test1, test2, result_path)