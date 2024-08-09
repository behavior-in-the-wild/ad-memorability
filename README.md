# Long-Term Ad Memorability: Understanding & Generating Memorable Ads



- [**Project Page**](https://behavior-in-the-wild.github.io/memorability.html)
- [**Data (LAMBDA)**](https://huggingface.co/datasets/behavior-in-the-wild/LAMBDA)
- [**Data (UltraLAMBDA)**](https://huggingface.co/datasets/behavior-in-the-wild/UltraLAMBDA)
- [**Paper**](https://arxiv.org/abs/2309.00378)

<div align="center">
    <img width="100%" src="imgs/example.png" alt="Example Image"/>
</div>

---

## Installation and Setup

Follow the steps below to install the required packages and set up the environment.

### Step 1: Clone the Repository

Open your terminal and clone the repository using the following command:

```shell
git clone https://github.com/behavior-in-the-wild/ad-memorability.git
```

### Step 2: Set Up the Conda Environment

Create and activate the Conda environment:

```shell

conda create -n admem python=3.10 -y
conda activate admem
pip install --upgrade pip  # Enable PEP 660 support
pip install -e .
pip install ninja
pip install flash-attn --no-build-isolation
```

### Step 3: Set Up Model Zoo

Create directories and download the required models:

```shell
mkdir model_zoo
mkdir model_zoo/LAVIS
cd ./model_zoo/LAVIS
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth
```
### Step 4: Set Up LLaMA-VID

```shell
cd path/to/ad-memorability
mkdir work_dirs
cd work_dirs
git lfs install
git clone https://huggingface.co/YanweiLi/llama-vid-13b-full-224-video-fps-1
```

### Step 5: Prepare Data Directory

```shell
cd path/to/ad-memorability
mkdir data
cd ./data
```
1. Create .npy files of your videos. A sample file is given in the sample folder. 
2. Store them in as ./data/videos/video_scenes/{id}.npy


## Training

Install your desired deepspeed version and run train.sh

```shell
bash train.sh
```

## Inference

1. For predicting memorability scores:
```shell
bash eval_bs.sh
```
2. For generating memorable videos:
```shell
bash eval_cs.sh
```

## Citation
If you find this repo useful for your research, please consider citing the paper


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


## Acknowledgement
We would like to thank the following repos for their great work:
- This work is built upon the [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID)
- This work is built upon the [LLaVA](https://github.com/haotian-liu/LLaVA).
- This work utilizes LLMs from [Vicuna](https://github.com/lm-sys/FastChat).
- This work utilizes pretrained weights from [InstructBLIP](https://github.com/salesforce/LAVIS).
- We perform video-based evaluation from [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT).

## License

The data and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA-VID,LLaVA, LLaMA, Vicuna and GPT-4.
