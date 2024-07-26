# Long-Term Ad Memorability: Understanding & Generating Memorable Ads

 [[Project Page](https://behavior-in-the-wild.github.io/memorability.html)]   [[Data](https://huggingface.co/datasets/behavior-in-the-wild/LAMBDA)] [[Paper](https://arxiv.org/abs/2309.00378)]


## Repository Links
The code uses the repositories:
- Ask Anything Repository: [https://github.com/OpenGVLab/Ask-Anything.git](https://github.com/OpenGVLab/Ask-Anything.git)
- Open-Assistant Repository: [https://github.com/LAION-AI/Open-Assistant.git](https://github.com/LAION-AI/Open-Assistant.git)

## Installation

Before running the code, you need to install the dependencies and set up the environments for both Ask Anything and Open-Assistant. Detailed installation instructions can be found in the respective READMEs:

- Ask Anything Installation Instructions: [https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat/README.md](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat/README.md)
- Open-Assistant Installation Instructions: [https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/README.md](https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/README.md)

Replace the model_training folder with the one provided in our paper's code.

## Running the Code

Follow the steps below to run the integrated code:

1. **Preprocess Frames (Extract Visual Embeddings)**

   To extract the visual embeddings for Lamem using Ask Anything, execute the following commands:

   ```shell
   cd src/Ask-Anything/videochat
   python run.py
   ```
2. **Model Training**

   To train the model, execute the following commands:

   ```shell
   cd src/Open-Assistant/model/model_training
   deepspeed trainer_sft.py --configs Lamem --wandb-entity <wandb> --deepspeed
   ```

   Replace <wandb> with the actual entity for WandB (if applicable).

3. **Model Inference**

   To run the model inference, execute the following commands:

   ```shell
   cd src/Ask-Anything/videochat
   python run_eval.py
   ```
Replace with appropriate paths in the code.


## Citation
If you find our work useful for your research and applications, please cite using this BibTeX:
```bibtex
@misc{s2024longtermadmemorabilityunderstanding,
            title={Long-Term Ad Memorability: Understanding and Generating Memorable Ads}, 
            author={Harini S I au2 and Somesh Singh and Yaman K Singla and Aanisha Bhattacharyya and Veeky Baths and Changyou Chen and Rajiv Ratn Shah and Balaji Krishnamurthy},
            year={2024},
            eprint={2309.00378},
            archivePrefix={arXiv},
            primaryClass={cs.CL},
            url={https://arxiv.org/abs/2309.00378}}
```