import yaml
import argparse
import torch
from peft import LoraConfig
from dataset import generate_phase_one_dataloader, generate_phase_two_dataloader
from model import PhiClipMLLM
from trainer import CustomTrainer
from transformers import TrainingArguments
from torch.optim import AdamW

def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def perform_training(args, cfg, peft_config):
    
    
    # Load dataloader based on phase
    if args.phase_one:
        mllm = PhiClipMLLM(
            language_model=cfg['model']['text_model'],
            vision_model=cfg['model']['vision_model'],
            adapter_dim=cfg['model']['adapter_dim'],
            dtype=torch.bfloat16 if cfg['dtype'] == 'torch.bfloat16' else torch.float32
        )

        print("Model loaded")
        print("Starting Phase One training ------ ")
        
        dataloader = generate_phase_one_dataloader(
            annotation_file=cfg['phase_one']['annotation_file'],
            image_dir=cfg['phase_one']['image_directory'],
            batch_size=cfg['training_args']['batch_size']
        )
    else:
        print("Starting Phase Two training ------ ")
        mllm = PhiClipMLLM.from_pretrained(checkpoint_path = cfg['checkpoint']['load_dir'], 
                                           dtype=torch.bfloat16 if cfg['dtype'] == 'torch.bfloat16' else torch.float32)
        print("what is happening")
        # mllm = PhiClipMLLM(
        #     language_model=cfg['model']['text_model'],
        #     vision_model=cfg['model']['vision_model'],
        #     adapter_dim=cfg['model']['adapter_dim'],
        #     dtype=torch.bfloat16 if cfg['dtype'] == 'torch.bfloat16' else torch.float32
        # )
        print(cfg['dtype'])
        mllm.setup_lora(peft_config)
        dataloader = generate_phase_two_dataloader(
            directory=cfg['phase_two']['directory'],
            batch_size=cfg['training_args']['batch_size']
        )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=cfg['checkpoint']['output_dir'],
        num_train_epochs=cfg['training_args']['num_train_epochs'],
        logging_dir="./logs",
        learning_rate=cfg['training_args']['learning_rate'],
        save_steps=cfg['training_args']['save_steps'],
        logging_steps=cfg['training_args'].get('logging_steps', 10),  # Default to 10 if not provided
        bf16=True if cfg['dtype'] == 'torch.bfloat16' else False,
        max_grad_norm=1.0,
    )
    optimizer = AdamW(mllm.parameters(), lr=cfg['training_args']['learning_rate'], weight_decay=0.001)
    # Trainer
    trainer = CustomTrainer(
        model=mllm,
        args=training_args,
        train_dataloader=dataloader,
        optimizers=(optimizer, None)
    )

    # Train
    trainer.train()
    trainer.save_model(cfg['checkpoint']['output_dir'])

def main():
    parser = argparse.ArgumentParser(description="Train a multimodal model.")
    parser.add_argument('--phase_one', action='store_true', help="Run phase one training.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load the YAML configuration
    cfg = load_config(args.config)

    # Initialize the model
    peft_config_text = None
    peft_config_vision = None
    peft_config_adapter = None
    if 'peft_text' in cfg and cfg['peft_text']:
        peft_config_text = LoraConfig(
            r=cfg['peft_text'].get('r', 16),
            lora_alpha=cfg['peft_text'].get('lora_alpha', 32),
            target_modules=cfg['peft_text'].get('target_modules', ["query", "value"]),
            lora_dropout=cfg['peft_text'].get('lora_dropout', 0.1),
            bias=cfg['peft_text'].get('bias', "none")
        )
    if 'peft_vision' in cfg and cfg['peft_vision']:
        peft_config_vision = LoraConfig(
            r=cfg['peft_vision'].get('r', 16),
            lora_alpha=cfg['peft_vision'].get('lora_alpha', 32),
            target_modules=cfg['peft_vision'].get('target_modules', ["query", "value"]),
            lora_dropout=cfg['peft_vision'].get('lora_dropout', 0.1),
            bias=cfg['peft_vision'].get('bias', "none")
        )
    if 'peft_adapter' in cfg and cfg['peft_adapter']:
        peft_config_adapter = LoraConfig(
            r=cfg['peft_adapter'].get('r', 16),
            lora_alpha=cfg['peft_adapter'].get('lora_alpha', 32),
            target_modules=cfg['peft_adapter'].get('target_modules', ["query", "value"]),
            lora_dropout=cfg['peft_adapter'].get('lora_dropout', 0.1),
            bias=cfg['peft_adapter'].get('peft_adapter', "none")
        )
    print("here")
    perform_training(args, cfg, [peft_config_text, peft_config_vision, peft_config_adapter])

if __name__ == "__main__":
    main()