import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from glob import glob
import json
import os
from evals.metrics import eval_clusters_length_independent
import numpy as np
from tqdm import tqdm

def get_save_dir(save_dir, resume=False):
    curr_versions = glob(f"{save_dir}/version_*", recursive=False)
    if resume:
        # Re-use an existing versioned dir (passed explicitly from run_train.sh)
        return save_dir
    if not curr_versions:
        return f"{save_dir}/version_1"
    else:
        latest_version = max(curr_versions, key=lambda x: int(x.split('_')[-1]))
        latest_version_num = int(latest_version.split('_')[-1])
        return f"{save_dir}/version_{latest_version_num + 1}"


def calculate_perplexity(quantized_indices, codebook_size):
    """
    Calculates the perplexity of codebook usage when each input sample
    is assigned a single codebook index.

    Args:
        quantized_indicies (torch.Tensor): A 1D tensor of shape [batch_size]
                                         containing the assigned codebook indices.
                                         Values should be in the range [0, num_embeddings-1].
        codebook_size (int): The total number of embeddings in the codebook.

    Returns:
        float: The calculated perplexity.
    """
    counts = torch.bincount(quantized_indices.long(), minlength=codebook_size)
    total_assignments = quantized_indices.numel()
    avg_probs = counts.float() / total_assignments
    avg_probs = avg_probs + 1e-10 # Add epsilon for numerical stability
    entropy = -torch.sum(avg_probs * torch.log(avg_probs))
    perplexity = torch.exp(entropy).item()
    return perplexity


class VQVAETrainer:
    def __init__(self, model, optimizer, train_loader, val_loader=None,
                 device='cpu', epochs=100, use_wandb=False,
                 save_dir=None, scheduler=None, warmup_epochs=0, resume=False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.num_epochs = epochs
        self.epoch = 1
        self.step = 1
        self.save_dir = save_dir
        self.use_wandb = use_wandb
        self.ckpt_loss = {}
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            self.ckpt_dir = os.path.join(self.save_dir, "checkpoints")
            os.makedirs(self.ckpt_dir, exist_ok=True)
            self.ckpt_loss_file = os.path.join(self.save_dir, "model_loss.txt")

            if resume:
                self._resume()
            else:
                with open(os.path.join(self.save_dir, "model_config.json"), 'w') as f:
                    json.dump(model.get_config(), f, indent=4)

            if use_wandb:
                import wandb
                wandb.config.update({"save_dir": self.save_dir}, allow_val_change=True)
        else:
            self.ckpt_dir = None
            self.ckpt_loss_file = None

    def _find_latest_checkpoint(self):
        """Return (path, epoch_number) of the most recent checkpoint, or (None, 0)."""
        checkpoints = glob(os.path.join(self.ckpt_dir, "model_epoch_*.pt"))
        if not checkpoints:
            return None, 0
        def _epoch_num(p):
            return int(os.path.basename(p).replace("model_epoch_", "").replace(".pt", ""))
        latest = max(checkpoints, key=_epoch_num)
        return latest, _epoch_num(latest)

    def _resume(self):
        """Load the latest checkpoint and restore training state."""
        ckpt_path, start_epoch = self._find_latest_checkpoint()
        if ckpt_path is None:
            # No checkpoints yet — fresh start, still need to write config
            with open(os.path.join(self.save_dir, "model_config.json"), 'w') as f:
                json.dump(self.model.get_config(), f, indent=4)
            print("No checkpoint found — starting from scratch.")
            return
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.epoch = start_epoch + 1
        self.step = start_epoch * len(self.train_loader) + 1
        # Restore loss history so model_loss.txt is accurate after resuming
        if os.path.exists(self.ckpt_loss_file):
            with open(self.ckpt_loss_file) as f:
                for line in f:
                    line = line.strip()
                    if ':' in line:
                        k, v = line.split(':', 1)
                        try:
                            self.ckpt_loss[k.strip()] = float(v.strip())
                        except ValueError:
                            pass
        print(f"Resumed from {ckpt_path}")
        print(f"Continuing from epoch {self.epoch}/{self.num_epochs}")
    
    def write_loss(self):
        sorted_loss = sorted(self.ckpt_loss.items(), key=lambda x: x[1])
        with open(self.ckpt_loss_file, 'w') as f:
            for key, value in sorted_loss:
                f.write(f"{key}: {value}\n")
            f.write("\n")

    def model_step(self, batch, val=False):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                if key == 'id':
                    continue
                batch[key] = batch[key].to(self.device)
        output = self.model(batch, val=val)
        return output
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_commitment_loss = 0
        total_codebook_loss = 0
        total_recon_loss = 0
        total_dihedral_loss = 0
        total_aa_loss = 0
        total_loop_length_loss = 0
        total_pred_loop_length_loss = 0
        quantized_indices_list = []
        pred_aa_list = []

        for batch in tqdm(self.train_loader, total=len(self.train_loader), desc=f"Training Epoch {self.epoch}"):
            output = self.model_step(batch)
            self.optimizer.zero_grad()
            output.loss.backward()
            self.optimizer.step()
            if self.use_wandb:
                import wandb
                wandb.log({'Train Loss': output.loss.item(), 'Epoch': self.epoch, 'Step': self.step, 'Reconstruction Loss': output.recon_loss.item(),
                        'Commitment Loss': output.commit_loss.item(), 'Codebook Loss': output.codebook_loss.item(), 'Dihedral Loss': output.dihedral_loss.item(),
                        'AA Loss': output.aa_loss.item()})
            self.step += 1
            total_loss += output.loss.item()
            total_commitment_loss += output.commit_loss.item()
            total_codebook_loss += output.codebook_loss.item()
            total_recon_loss += output.recon_loss.item()
            total_dihedral_loss += output.dihedral_loss.item()
            total_aa_loss += output.aa_loss.item()
            total_loop_length_loss += output.loop_length_loss.item()
            total_pred_loop_length_loss += output.pred_loop_length_loss.item()
            quantized_indices_list.append(output.quantized_indices.detach().cpu())
            pred_aa_list.append(output.pred_aa.detach().cpu())

        perplexity = calculate_perplexity(torch.cat(quantized_indices_list), self.model.codebook_size)
        epoch_loss = total_loss / len(self.train_loader)
        epoch_commit_loss = total_commitment_loss / len(self.train_loader)
        epoch_codebook_loss = total_codebook_loss / len(self.train_loader)
        epoch_recon_loss = total_recon_loss / len(self.train_loader)
        epoch_dihedral_loss = total_dihedral_loss / len(self.train_loader)
        epoch_aa_loss = total_aa_loss / len(self.train_loader)
        epoch_loop_length_loss = total_loop_length_loss / len(self.train_loader)
        epoch_pred_loop_length_loss = total_pred_loop_length_loss / len(self.train_loader)

        if self.save_dir:
            model_state_dict = self.model.state_dict()
            for key in model_state_dict:
                if torch.is_tensor(model_state_dict[key]):
                    model_state_dict[key] = model_state_dict[key].cpu()
            torch.save(model_state_dict, os.path.join(self.ckpt_dir, f"model_epoch_{self.epoch}.pt"))

        return epoch_loss, epoch_commit_loss, epoch_codebook_loss, epoch_recon_loss, epoch_dihedral_loss, epoch_aa_loss, epoch_loop_length_loss, epoch_pred_loop_length_loss, perplexity

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        total_commit_loss = 0
        total_codebook_loss = 0
        total_recon_loss = 0
        total_dihedral_loss = 0
        total_aa_loss = 0
        total_loop_length_loss = 0
        total_pred_loop_length_loss = 0
        quantized_indices_list = []
        gt_aa_list = []
        pred_aa_list = []

        with torch.no_grad():
            for batch in self.val_loader:
                output = self.model_step(batch, val=True)
                gt_aa_list.append(output.true_aa.flatten().numpy())
                total_loss += output.loss.item()
                total_commit_loss += output.commit_loss.item()
                total_codebook_loss += output.codebook_loss.item()
                total_recon_loss += output.recon_loss.item()
                total_dihedral_loss += output.dihedral_loss.item()
                total_aa_loss += output.aa_loss.item()
                total_loop_length_loss += output.loop_length_loss.item()
                total_pred_loop_length_loss += output.pred_loop_length_loss.item()
                quantized_indices_list.append(output.quantized_indices.detach().cpu())
                pred_aa_list.append(output.pred_aa.argmax(dim=-1).flatten().numpy())

        quantized_indices_list = torch.cat(quantized_indices_list)
        quantized_indices_list_numpy = quantized_indices_list.numpy()

        pred_aa_list = np.concatenate(pred_aa_list)
        gt_aa_list = np.concatenate(gt_aa_list)
        aa_recovery = np.mean(pred_aa_list == gt_aa_list)
        print(f"Validation AA Recovery: {aa_recovery:.4g}")

        all_angles = np.array([self.val_loader.dataset[i]['angles'] for i in range(len(self.val_loader.dataset))])
        all_loop_coords = np.array([self.val_loader.dataset[i]['loop_c_alpha_coords'] for i in range(len(self.val_loader.dataset))])
        all_stem_coords = np.array([self.val_loader.dataset[i]['stem_c_alpha_coords'] for i in range(len(self.val_loader.dataset))])
        all_tokens = np.array([self.val_loader.dataset[i]['sequence'] for i in range(len(self.val_loader.dataset))])

        special_tokens_mask = (
            (all_tokens == self.model.encoder.alphabet.cls_idx) | (all_tokens == self.model.encoder.alphabet.eos_idx) | (all_tokens == self.model.encoder.alphabet.padding_idx)
        )  # B, T
        proportion_correct, phi, psi, omega = eval_clusters_length_independent(
            all_angles, all_loop_coords, all_stem_coords, quantized_indices_list_numpy, ~special_tokens_mask, run_alignment=False)
        
        num_clusters_with_different_lengths = 0
        max_length_diff = 0
        for cluster in np.unique(quantized_indices_list_numpy):
            cluster_indices = np.where(quantized_indices_list_numpy == cluster)[0]
            cluster_lengths = ~special_tokens_mask[cluster_indices].sum(axis=1)
            if len(np.unique(cluster_lengths)) > 1:
                num_clusters_with_different_lengths += 1
            max_length_diff = max(max_length_diff, np.max(cluster_lengths) - np.min(cluster_lengths))
        print(f"Validation: Number of clusters with different lengths: {num_clusters_with_different_lengths}, Max length difference: {max_length_diff}")
        print(f"Validation: proportion correct: {proportion_correct:.4g}, phi: {phi:.4g}, psi: {psi:.4g}, omega: {omega:.4g}")

        if self.use_wandb:
            import wandb
            wandb.log({'Epoch Validation Proportion Correct': proportion_correct, 'Epoch Validation Phi': phi, 'Epoch Validation Psi': psi, 'Epoch Validation Omega': omega,
                       'Epoch Validation num used codebook indices': len(torch.unique(quantized_indices_list)), 'Epoch Validation AA Recovery': aa_recovery,
                       'Epoch Validation num clusters with different lengths': num_clusters_with_different_lengths,
                       'Epoch Validation Max Length Difference': max_length_diff})

        perplexity = calculate_perplexity(quantized_indices_list, self.model.codebook_size)

        epoch_val_loss = total_loss / len(self.val_loader)
        epoch_commit_loss = total_commit_loss / len(self.val_loader)
        epoch_codebook_loss = total_codebook_loss / len(self.val_loader)
        epoch_recon_loss = total_recon_loss / len(self.val_loader)
        epoch_dihedral_loss = total_dihedral_loss / len(self.val_loader)
        epoch_aa_loss = total_aa_loss / len(self.val_loader)
        epoch_loop_length_loss = total_loop_length_loss / len(self.val_loader)
        epoch_pred_loop_length_loss = total_pred_loop_length_loss / len(self.val_loader)

        # update the checkpoint loss dictionary
        loss_key = f"model_epoch_{self.epoch}.pt"
        self.ckpt_loss[loss_key] = epoch_val_loss
        if self.save_dir:
            self.write_loss()

        return epoch_val_loss, epoch_commit_loss, epoch_codebook_loss, epoch_recon_loss, epoch_dihedral_loss, epoch_aa_loss, epoch_loop_length_loss, epoch_pred_loop_length_loss, perplexity

    def train(self):
        while self.epoch <= self.num_epochs:
            train_loss, train_commit_loss, train_codebook_loss, train_recon_loss, total_dihedral_loss, train_aa_loss, train_loop_length_loss, train_pred_loop_length_loss, train_perplexity = self.train_epoch()
            if self.scheduler and epoch > self.warmup_epochs:
                self.scheduler.step()
                print(f"Learning rate at epoch {epoch}: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f'Epoch {epoch}/{self.num_epochs}, Train Loss: {train_loss:.4g}, Train Commitment Loss: {train_commit_loss:.4g}, '
                  f'Train Codebook Loss: {train_codebook_loss:.4g}, Train Reconstruction Loss: {train_recon_loss:.4g}, '
                  f'Train Dihedral Loss: {total_dihedral_loss:.4g}, Train AA Loss: {train_aa_loss:.4g}, '
                  f'Train Loop Length Loss: {train_loop_length_loss:.4g}, Train Pred Loop Length Loss: {train_pred_loop_length_loss:.4g}, Train Perplexity: {train_perplexity:.4g}')
            if self.use_wandb:
                import wandb
                wandb.log({'Epoch Train Loss': train_loss, 'Epoch': self.epoch, 'Epoch Train Commitment Loss': train_commit_loss,
                            'Epoch Train Codebook Loss': train_codebook_loss, 'Epoch Train Reconstruction Loss': train_recon_loss,
                            'Epoch Train Perplexity': train_perplexity, 'Epoch Train Dihedral Loss': total_dihedral_loss,
                            'Epoch Train AA Loss': train_aa_loss, 'Epoch Train Loop Length Loss': train_loop_length_loss,
                            'Epoch Train Pred Loop Length Loss': train_pred_loop_length_loss})
            if self.val_loader is not None:
                val_loss, val_commit_loss, val_codebook_loss, val_recon_loss, val_dihedral_loss, val_aa_loss, val_loop_length_loss, val_pred_loop_length_loss, val_perplexity = self.validate_epoch()
                print(f'Epoch {epoch}/{self.num_epochs}, Validation Loss: {val_loss:.4g}, Validation Commitment Loss: {val_commit_loss:.4g}, '
                      f'Validation Codebook Loss: {val_codebook_loss:.4g}, Validation Reconstruction Loss: {val_recon_loss:.4g}, '
                      f'Validation Dihedral Loss: {val_dihedral_loss:.4g}, Validation AA Loss: {val_aa_loss:.4g}, '
                      f'Validation Loop Length Loss: {val_loop_length_loss:4g}, Validation Pred Loop Length Loss: {val_pred_loop_length_loss:4g}, Validation Perplexity: {val_perplexity:.4g}')
                if self.use_wandb:
                    import wandb
                    wandb.log({'Epoch Validation Loss': val_loss, 'Epoch Validation Commitment Loss': val_commit_loss,
                               'Epoch Validation Codebook Loss': val_codebook_loss, 'Epoch Validation Reconstruction Loss': val_recon_loss,
                               'Epoch Validation Dihedral Loss': val_dihedral_loss, 'Epoch Validation AA Loss': val_aa_loss,
                               'Epoch Validation Loop Length Loss': val_loop_length_loss, 'Epoch Validation Pred Loop Length Loss': val_pred_loop_length_loss,
                               'Epoch Validation Perplexity': val_perplexity})
            self.epoch += 1
        if self.use_wandb:
            import wandb
            wandb.finish()