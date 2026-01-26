# csc-52002-notes
My personal notes for CSC-52002: Multimodal Artificial Intelligence at École Polytechnique (Winter 2025-2026).

## Schedule
### 1. Introduction to Multimodal AI & Tasks
- Definition of multimodal learning (vision, language, audio)
- Computer Vision tasks:
    - Image classification
    - Object detection
    - Semantic segmentation
    - Instance segmentation
    - Pose estimation
- Video understanding:
    - Video classification
    - Action detection
    - Spatio-temporal localization
- Vision-language tasks:
    - Image captioning
    - Visual Question Answering (VQA)
- Image synthesis and style transfer
- Applications (safety, healthcare, accessibility, entertainment)

#### References
- K. He et al., *Mask R-CNN*, ICCV 2017
- I. Goodfellow et al., *Generative Adversarial Networks*, NeurIPS 2014
- P. Isola et al., *Image-to-Image Translation with Conditional GANs*, CVPR 2017
- J.-Y. Zhu et al., *CycleGAN*, ICCV 2017
- R. Girshick et al., *Faster R-CNN*, NeurIPS 2015

#### Related Resources
- [The Illustrated Bert (article)](https://jalammar.github.io/illustrated-bert/)
- [The Illustrated DeepSeek-R1 (article)](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1)
- [The Illustrated GPT-OSS (article)](https://substack.com/@jayalammar/p-170260890)

### 2. Evolution of Multimodal Generative Models & Architectures
- Timeline of generative models:
    - GAN (2014)
    - ProGAN (2017)
    - Transformers (2017)
    - DDPM (2020)
    - VQ-GAN (2021)
    - DALLE / GLIDE (2022)
    - Stable Diffusion (2022)
- Two dominants paradigms:
    - Transformers (text, tokens, code)
    - Stable diffusion (image, audio, continuous data)

#### References
- I. Goodfellow et al., *GANs*, NeurIPS 2014
- A. Radford et al., *DCGAN*, ICLR 2016
- A. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
- J. Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020
- P. Esser et al., *VQ-GAN*, CVPR 2021
- R. Ramesh et al., *DALL·E*, ICML 2021
- A. Nichol et al., *GLIDE*, ICML 2022
- [Kaplan scaling laws paper](https://arxiv.org/abs/2001.08361)

flamingo paper... somewhere, maybe not here exactly

#### Related Resources
- [The Illustrated Transformer (article)](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer (article)](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Dive Into Deep Learning Chapter 11: Attention Mechanisms and Transformers](https://www.d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)
- [Chinchilla's Wild Implications (article)](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-scaling-laws)
- [Contrastive Representation Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/)
- [Generalized Visual Language Models](https://lilianweng.github.io/posts/2022-06-09-vlm/)
- [Introduction to Transformers w/ Andrej Karpathy (video)](https://www.youtube.com/watch?v=XfpMkf4rD6E)

### 3. Multimodal Large Language Models & Vision-Language Models
- LLM Basics and transformer self-attention
- Encoder-only, decoder-only, encoder–decoder models
- Vision–Language Tasks:
  - Visual reasoning
  - Visual grounding
  - VQA
  - Image captioning
- Early VLM architectures:
  - Two-stream models
  - Single-stream models
- Families of VLMs:
  - Contrastive
  - Masked modeling
  - Generative
  - Perception + LLM hybrids

#### References
- J. Lu et al., *ViLBERT*, NeurIPS 2019
- H. Tan & M. Bansal, *LXMERT*, EMNLP 2019
- L. Li et al., *VisualBERT*, arXiv 2019
- Y.-C. Chen et al., *UNITER*, ECCV 2020
- A. Radford et al., *CLIP*, ICML 2021
- J. Alayrac et al., *Flamingo*, NeurIPS 2022

#### Related Resources
- [Multimodal Neurons in Artificial Neural Networks (article)](https://distill.pub/2021/multimodal-neurons/)
- [CLIP: Connecting text and images](https://openai.com/index/clip/)


### 4. Generative Models: GANs, VAEs, Diffusion
- Relationship between discriminative, generative, conditional models
- GAN formulation:
  - Generator vs discriminator
  - Minimax objective
  - Mode collapse
  - Training instability
- DCGAN design rules
- Conditional GANs and image translation
- Diffusion models as likelihood-based generative models

#### References
- I. Goodfellow et al., *GANs*, NeurIPS 2014
- D. Kingma & M. Welling, *Auto-Encoding Variational Bayes*, ICLR 2014
- A. Radford et al., *DCGAN*, ICLR 2016
- T. Karras et al., *StyleGAN*, CVPR 2019
- J. Ho et al., *DDPM*, NeurIPS 2020
- R. Rombach et al., *Latent Diffusion Models*, CVPR 2022

#### Related Resources
- [The Annotated Diffusion Model (article)](https://huggingface.co/blog/annotated-diffusion)
- [The Illustrated Stable Diffusion (article)](https://jalammar.github.io/illustrated-stable-diffusion/)
- [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/)
- [CS231n Lecture 13: Generative Models 1](https://www.youtube.com/watch?v=zbHXQRUNlH0&list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16&index=13)
- [CS231n Lecture 14: Generative Models 2](https://www.youtube.com/watch?v=Edr4uZFh4EE&list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16&index=14)
- [Dive Into Deep Learning Chapter 20: Generative Adversarial Networks](https://d2l.ai/chapter_generative-adversarial-networks/index.html)
- [What are Diffusion Models (article)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

## Books
- [Understanding Deep Learning (Prince)](https://udlbook.github.io/udlbook/)
- [Deep Learning (Goodfellow)](https://www.deeplearningbook.org)
- [Pattern Recognition and Machine Learning (Bishop)](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
- [Dive into Deep Learning](https://d2l.ai)
- [Computer Vision: Algorithms and Applications (Szeliski)](https://szeliski.org/Book/)

## Related Courses
- [CS231n](https://cs231n.stanford.edu)
- [CS336](https://stanford-cs336.github.io/spring2025/)
- [CS236](https://deepgenerativemodels.github.io)
- [Hugging Face Courses](https://huggingface.co/learn)
- [CS224n]
- CS229
- CS230
- [CS25: Transformers United](https://web.stanford.edu/class/cs25/)
- [Neural Networks: Zero to Hero (course)](https://karpathy.ai/zero-to-hero.html)

## Miscellaneous
- [Lilian Weng (blog)](https://lilianweng.github.io)
- [Andrej Karpathy (blog)](https://karpathy.ai/#:~:text=I%20have%20three%20blogs%20🤦%E2%80%8D♂%EF%B8%8F.%20This%20GitHub%20blog%20is%20my%20oldest%20one.%20I%20then%20briefly%20and%20sadly%20switched%20to%20my%20second%20blog%20on%20Medium.%20I%20now%20have%20a%20Bear%20blog.)
- [Jason Wei (blog)](https://www.jasonwei.net/blog)
- [PyTorch Intro](https://docs.pytorch.org/tutorials/intro.html), [PyTorch Recipes](https://docs.pytorch.org/tutorials/recipes_index.html)
- [Distill (blog)](https://distill.pub)
- [Learning from Text (video)](https://www.youtube.com/watch?v=BnpB3GrpsfM)

<!--
## Papers

### Foundations (Vision, NLP, Representation)
- Y. LeCun et al., *Gradient-Based Learning Applied to Document Recognition*, IEEE 1998
- A. Krizhevsky et al., *ImageNet Classification with Deep CNNs*, NeurIPS 2012
- K. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016
- T. Mikolov et al., *Distributed Representations of Words and Phrases*, NeurIPS 2013
- J. Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers*, NAACL 2019

---

### Vision–Language Models (VLMs)
- J. Lu et al., *ViLBERT*, NeurIPS 2019
- H. Tan & M. Bansal, *LXMERT*, EMNLP 2019
- L. Li et al., *VisualBERT*, arXiv 2019
- Y.-C. Chen et al., *UNITER*, ECCV 2020
- A. Radford et al., *CLIP*, ICML 2021
- J. Alayrac et al., *Flamingo*, NeurIPS 2022
- J. Yu et al., *CoCa: Contrastive Captioners*, NeurIPS 2022
- J. Li et al., *BLIP*, ICML 2022
- J. Li et al., *BLIP-2*, ICML 2023

---

### Multimodal LLMs & Unified Models
- R. Ramesh et al., *DALL·E*, ICML 2021
- H. Liu et al., *LLaVA*, NeurIPS 2023
- Microsoft, *Kosmos-1*, ICLR 2023
- Meta AI, *ImageBind*, CVPR 2023
- OpenAI, *GPT-4 Technical Report*, 2023
- DeepMind, *Gemini*, 2023

---

### Generative Models – GANs
- I. Goodfellow et al., *Generative Adversarial Nets*, NeurIPS 2014
- A. Radford et al., *DCGAN*, ICLR 2016
- P. Isola et al., *pix2pix*, CVPR 2017
- J.-Y. Zhu et al., *CycleGAN*, ICCV 2017
- T. Karras et al., *Progressive Growing of GANs*, ICLR 2018
- T. Karras et al., *StyleGAN*, CVPR 2019
- T. Karras et al., *StyleGAN2*, CVPR 2020

---

### Generative Models – VAEs
- D. Kingma & M. Welling, *Auto-Encoding Variational Bayes*, ICLR 2014
- D. Rezende et al., *Stochastic Backpropagation*, ICML 2014
- C. Doersch, *Tutorial on Variational Autoencoders*, arXiv 2016
- I. Higgins et al., *β-VAE*, ICLR 2017

---

### Generative Models – Diffusion
- J. Sohl-Dickstein et al., *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*, ICML 2015
- J. Ho et al., *DDPM*, NeurIPS 2020
- Y. Song et al., *Score-Based Generative Modeling*, ICLR 2021
- A. Nichol & P. Dhariwal, *Improved DDPM*, ICML 2021
- R. Rombach et al., *Latent Diffusion Models*, CVPR 2022
- C. Saharia et al., *Imagen*, ICML 2022

---

### Scaling Laws & Training Dynamics
- J. Kaplan et al., *Scaling Laws for Neural Language Models*, arXiv 2020
- T. Hoffmann et al., *Training Compute-Optimal LLMs (Chinchilla)*, NeurIPS 2022
- H. Touvron et al., *LLaMA*, ICML 2023
- OpenAI, *GPT-3*, NeurIPS 2020

---

### Representation & Contrastive Learning
- T. Chen et al., *SimCLR*, ICML 2020
- K. He et al., *MoCo*, CVPR 2020
- J.-B. Grill et al., *BYOL*, NeurIPS 2020
- A. Radford et al., *CLIP*, ICML 2021

---

### Interpretability & Understanding Models
- C. Olah et al., *Feature Visualization*, Distill 2017
- C. Olah et al., *The Building Blocks of Interpretability*, Distill 2018
- Anthropic, *Transformer Circuits*, 2021–2023 -->