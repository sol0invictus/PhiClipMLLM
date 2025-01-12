import os

from transformers import Trainer


class CustomTrainer(Trainer):
    def __init__(self, *args, train_dataloader=None, eval_dataloader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_dataloader = train_dataloader
        self._eval_dataloader = eval_dataloader

    def get_train_dataloader(self):
        return (
            self._train_dataloader
            if self._train_dataloader
            else super().get_train_dataloader()
        )

    def get_eval_dataloader(self, eval_dataset=None):
        return (
            self._eval_dataloader
            if self._eval_dataloader
            else super().get_eval_dataloader(eval_dataset)
        )

    def save_model(self, output_dir=None, _internal_call=True):
        try:
            # Ensure the output directory exists with full permissions
            os.makedirs(output_dir, exist_ok=True)

            # Check write permissions
            if output_dir is None or not os.access(output_dir, os.W_OK):
                print(f"Warning: No write permissions for {output_dir}")
                return

            # Only save on the main process to prevent multiple processes from interfering
            if not _internal_call and not self.is_world_process_zero():
                return

            # Save the model weights
            print(f"Saving model to {output_dir}")

            if output_dir is not None:
                self.model.save_checkpoint(
                    os.path.join(output_dir, "model_checkpoint.pt")
                )
            else:
                print("Error: output_dir is None")

        except PermissionError:
            print(f"Permission denied when trying to save to {output_dir}")
        except Exception as e:
            print(f"Error saving model: {e}")
