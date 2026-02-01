import torch

def save_training_checkpoint(path,
                             epoch,
                             global_step,
                             model,
                             optimizer,
                             scaler,
                             ema,
                             config
                             ):
    
    """
    Saves a checkpoint containing all training states.
    """

    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scale_state_dict": scaler.state_dict(),
        "ema_model_state_dict": ema.ema_model.state_dict(),
        "ema_step": ema.step,
        "config": config
    }, path)
    
    print(f"Checkpoint saved at {path}")