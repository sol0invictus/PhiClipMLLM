# PhiClipMLLM

PhiClipMLLM is a multimodal language model that integrates text and vision models to generate text based on input text and optional images. This project uses LoRA (Low-Rank Adaptation) for efficient fine-tuning.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training](#training)
- [Generating Text](#generating-text)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/PhiClipMLLM.git
    cd PhiClipMLLM
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Configuration

The configuration files are located in the [training_configs](http://_vscodecontentref_/1) directory. There are two main configuration files:

- [phi_clip.yaml](http://_vscodecontentref_/2): Configuration for the first phase of training.
- [phi_clip_p2.yaml](http://_vscodecontentref_/3): Configuration for the second phase of training.

### Training

To start training, use the [train.py](http://_vscodecontentref_/4) script with the appropriate configuration file.

#### Phase One Training

```bash
python train.py --phase_one --config training_configs/phi_clip.yaml