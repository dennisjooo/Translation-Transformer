import logging
from dotenv import load_dotenv

from config import config
from src.args import parse_args
from src.setup import (
    setup_directories, create_model, setup_lightning_model,
    setup_data_module, setup_wandb, setup_callbacks, setup_trainer
)
from src.utils import validate_config, save_model, handle_training_error

# Initialize environment
load_dotenv()

def main() -> None:
    """Main training function."""
    try:
        # Parse arguments and setup
        args = parse_args()
        setup_directories()
        config.update(vars(args))
        validate_config(config)
        
        # Create model and data module
        model = create_model(config)
        data_module = setup_data_module(config)  # Create data_module first
        
        # Setup training components
        lightning_model = setup_lightning_model(model, data_module, config)  # Pass data_module
        wandb_logger = setup_wandb(model, config)
        callbacks = setup_callbacks(model, data_module, config)
        trainer = setup_trainer(config, callbacks, wandb_logger)

        # Train the model
        trainer.fit(lightning_model, data_module)
        save_model(model, trainer, config)

    except Exception as e:
        handle_training_error(e)

if __name__ == "__main__":
    main()
