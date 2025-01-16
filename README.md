# PhiClipMLLM

PhiClipMLLM is a project that demonstrates building a simple MultiModal LLM from scratch, focusing on image and text processing. Following the design patterns of LLAVA and MinGPT4, this implementation leverages Huggingface's transformer ecosystem while providing clear visibility into the core MLLM architecture.

The project combines lightweight vision and language models to showcase efficient multimodal learning with minimal resources. It uses Microsoft's Phi-3-mini-4k-instruct for text processing and OpenAI's CLIP-ViT-Large-Patch14 for vision tasks. Through LoRA (Low-Rank Adaptation), the model enables fine-tuning with modest computational requirements, making it accessible to researchers and developers with limited hardware.

I have only tested it on 3090Ti.

*Note: This project is intended as a demonstration of training concepts. The training parameters are not optimized for maximum performance.*

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sol0invictus/PhiClipMLLM
cd PhiClipMLLM
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

This project uses the COCO 2017 dataset, a standard benchmark in computer vision tasks. You'll need to download two components:

1. Training images: [COCO 2017 Train Images](http://images.cocodataset.org/zips/train2017.zip)
2. Annotations: [COCO 2017 Stuff Annotations](http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip)

Download both files and store them in an appropriate directory. The project's dataloader is configured for the COCO 2017 dataset structure and annotation format.

## Configuration

Configuration files are stored in the `training_configs` directory and include:

* `phi_clip.yaml`: First phase training configuration
* `phi_clip_p2.yaml`: Second phase training configuration

While both configurations use the same dataset, they maintain separate files due to different optimizer settings.

## Training

The training process consists of two phases, managed by the `train.py` script:

### Phase One Training

```bash
python train.py --phase_one --config training_configs/phi_clip.yaml
```

During this phase, both vision and language backbones remain frozen while only the adapter undergoes training.

### Phase Two Training

```bash
python train.py --config training_configs/phi_clip_p2.yaml
```

Phase two involves fine-tuning the entire architecture. LoRA training is applied to vision and language backbones to reduce computational overhead.

## Text Generation

Use the `generate` method from the `model.py` package to generate text:

```python
from model import PhiClipMLLM
mllm = PhiClipMLLM.from_pretrained(r"model_checkpoint.pt", dtype=torch.bfloat16)
text = generate(a, ["Describe this image <|placeholder1|> "], images=[r"PXL_20241113_020744498.jpg"])
```

Use `<|placeholder1|>` as the image placeholder token in your prompts.
