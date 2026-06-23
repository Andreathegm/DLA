import os
import torch
import hydra
from tqdm import tqdm
from collections import defaultdict

class CustomTrainer:
    def __init__(self, model, train_loader, val_loader, args, logger, compute_metrics):
        
        ## first we take the train cfg
        cfg_trainer = args.trainer


        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader 
        self.args = args
        self.logger = logger
        self.compute_metrics = compute_metrics

        ## first we take the train cfg
        cfg_trainer = self.args.trainer
        
        self.device = torch.device(cfg_trainer.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.model.to(self.device)
        self.epochs = cfg_trainer.epochs
        self.log_every_n_steps = cfg_trainer.log_every_n_steps
        self.checkpoint_dir = cfg_trainer.checkpoint_dir
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.global_step = 0

        optimizer_partial = hydra.utils.instantiate(self.args.optimizer)
        self.optimizer = optimizer_partial(params=self.model.parameters())
        
        self.criterion = hydra.utils.instantiate(self.args.loss)

    def train(self):
        print("Starting training...")
        self.estimate_VRAM_usage()

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            total_samples = 0
            
            # Using tqdm for an elegant progress bar
            progress_bar = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.epochs}]")
            
            for x, y in progress_bar:
                x, y = x.to(self.device), y.to(self.device)
                batch_size = x.size(0)
                
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * batch_size
                total_samples += batch_size
                self.global_step += 1

                if self.global_step % self.log_every_n_steps == 0:
                    step_metrics = self.compute_metrics(outputs, y)
                    step_metrics["train/step_loss"] = loss.item()
                    self.logger.log(step_metrics, step=self.global_step)

                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = epoch_loss / total_samples
            self.logger.log({"train/epoch_loss": avg_loss, "epoch": epoch}, step=self.global_step)
            print(f"Epoch [{epoch+1}/{self.epochs}] | Train Loss: {avg_loss:.4f}")
            
            val_loss = self.evaluate_val()
            self.save_checkpoint(epoch, val_loss)

    def _evaluate(self, loader, prefix="val"):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        accumulated_metrics = defaultdict(list)
        
        with torch.no_grad():
            for x, y in tqdm(loader, desc=f"Evaluating ({prefix})"):
                x, y = x.to(self.device), y.to(self.device)
                batch_size = x.size(0)
                
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                batch_metrics = self.compute_metrics(outputs, y)
                for key, value in batch_metrics.items():
                    accumulated_metrics[key].append(value * batch_size)
                    
        avg_loss = total_loss / total_samples
        final_metrics = {f"{prefix}_loss": avg_loss}
        
        for key, list_values in accumulated_metrics.items():
            final_metrics[f"{prefix}_{key}"] = sum(list_values) / total_samples
            
        self.logger.log(final_metrics, step=self.global_step)
        print(f"[{prefix.upper()}] Loss: {avg_loss:.4f}")
        
        return avg_loss

    def evaluate_val(self):
        return self._evaluate(self.val_loader, prefix="val")
    
    def test(self, test_loader , weights_path=None):
        if weights_path is not None:
            self._load_weights(weights_path)
        return self._evaluate(test_loader, prefix="test") 
    
    def save_checkpoint(self, epoch, val_loss):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint restored from epoch {epoch+1}")
        return epoch

    def _load_weights(self, weights_path):
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        print(f"Weights loaded from {weights_path}")
    
    def estimate_VRAM_usage(self):
        params = sum(p.numel() for p in self.model.parameters())
        bytes_per_param = 4
        
        # Training from scratch with Adam requires storing weights, gradients, and 2 optimizer states 
        # which means multiplying the raw parameter size by roughly 4
        param_memory_bytes = params * bytes_per_param * 4
        param_memory_gb = param_memory_bytes / (1024 ** 3)
        
        print("\n--- VRAM Analysis ---")
        print(f"Total Parameters: {params:,}")
        print(f"Estimated Theoretical VRAM (Model + Adam Optimizer): ~{param_memory_gb:.2f} GB")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
            reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
            print(f"Currently Allocated VRAM: {allocated:.2f} GB")
            print(f"Currently Reserved VRAM: {reserved:.2f} GB")
        print("--------------------\n")