
数据
Synthetic Dysarthric Speech: A Supplement, Not a Substitute for Authentic Data in Dysarthric Speech Recognition
interspeech 2025、
合成构音数据（TTDS/VC）存在过度平滑/缺乏类内变异性问题，会导致模型学习到错误的规律性偏差。实验表明，合成数据仅适合作为预训练底座以提升鲁棒性，但绝不可替代真实患者数据进行最终对齐，混合训练，提升识别率的最佳路径。
https://www.isca-archive.org/interspeech_2025/li25n_interspeech.pdf

Training Data Augmentation for Dysarthric Automatic Speech Recognition by Text-to-Dysarthric-Speech Synthesis
interspeech 2024
建立构音数据工厂，利用f5tts/CosyVoice低步数推理合成含糊语音，通过One-Shot音色迁移解决无数据冷启动。
https://arxiv.org/abs/2406.08568
CDSD：Chinese Dysarthria Speech Database
interspeech 2023
数据：
AISHELL-1 语料库：该语料库涵盖多种领域，具备丰富的语言多样性，有助于训练泛化能力更强的语音识别模型。
小学和中学课文：部分受试者为 儿童，为了更符合儿童的朗读习惯，研究人员从 网络爬取了小学和中学演讲稿，补充到文本池中。
大规模中文构音障碍数据集， 普通话语音识别模型难以泛化到构音障碍语音，个性化微调可以显著提升识别准确率， 但是不同构音障碍个体之间难以泛化。
https://arxiv.org/pdf/2310.15930

Improving the Efficiency of Dysarthria Voice Conversion System Based on Data Augmentation
IEEE TNSR
https://ieeexplore.ieee.org/document/10313325
  CycleGAN /Diff-GAN /StarGAN-VC 模型将合成的正常语音转化为类构音障碍语料。StarGAN-VC 采用多对多说话人语音转换架构，其核心优势在于无需配对的源 - 目标语料即可完成训练
  少量目标构音障碍患者的真实语音，以及数据增强阶段生成的大量类构音障碍语音。这种 “真实 + 合成” 的混合数据模式，既保证了数据与目标患者的相关性多阶段+Gan
Accurate synthesis of Dysarthric Speech for ASR data augmentation
Speech Communication
https://www.sciencedirect.com/science/article/abs/pii/S0167639324000839
在第一阶段加入了严重程度系数/停顿插入模型 
Severity-Controlled FastSpeech 2 (Acoustic Model) + HiFi-GAN (Vocoder)。

Improved ASR Performance for Dysarthric Speech Using Two-stage Data Augmentation
静态降速模拟病理语速，动态魔改SpecAugment（加入复制/高频噪）模拟口吃与气息，极低成本增强模型鲁棒性。
Stutter Mask : 在频谱上随机复制几帧（模仿卡顿）。
Hypernasal Mask : 对高频/低频能量做特定衰减（模仿鼻音过重）。
Breathiness Mask : 注入随机高斯噪声（模仿漏气）
interspeech 2022
https://www.sciencedirect.com/science/article/pii/S0010482525003051


A Study on Model Training Strategies for Speaker-Independent and Vocabulary-Mismatched Dysarthric Speech Recognition
MDPI
https://www.mdpi.com/2076-3417/15/4/2006

SYNTHESIS OF NEW WORDS FOR IMPROVED DYSARTHRIC SPEECH RECOGNITION ON AN EXPANDED VOCABULARY
ICASSP 2021 
对于数据样本选择，数据样本划分很有意义，已见词/未见词区分
https://ieeexplore.ieee.org/abstract/document/9414869


ASR
Comparison of Acoustic and Textual Features for Dysarthria Severity Classification in Amyotrophic Lateral Sclerosis
interspeech 2025
融合声学特征分析，ASR转写的文本特征（词性/句法分析），构建多模态严重度分类器，声音不准的或者也可能语言组织能力退化， 可以帮助从多维度提供建议
https://www.isca-archive.org/interspeech_2025/ys25_interspeech.pdf

Data Augmentation using Speech Synthesis for Speaker-Independent Dysarthria Severity Classification
interspeech 2025
可以利用可控TTS合成不同严重等级的构音障碍语音，构建数据增强训练集，解决真实病理分级数据稀缺问题，提升严重度分类器在未见说话人上的泛化能力，采用逆严重度加权的数据混合策略。针对高CER的重度样本，提高合成数据占比（如3:1），对轻度样本降低占比；并引入课程学习，训练后期逐步剔除合成数据，迫使模型适配真实病理特征。
https://www.isca-archive.org/interspeech_2025/kim25w_interspeech.pdf

Bridging ASR and LLMs for Dysarthric Speech Recognition: Benchmarking Self-Supervised and Generative Approaches
interspeech 2025
利用LLM的上下文推理能力对ASR输出的N-best候选列表进行重排序与修复，针对Q-Former信息压缩导致细节丢失的问题，是否可以改用线性投影保留更多时序特征；同时采用全链路LoRA策略，在Whisper端修正声学特征，在LLM端适配病理语义，解决泛化性瓶颈。
https://arxiv.org/abs/2508.08027

Perceiver-Prompt: Flexible Speaker Adaptation in Whisper for Chinese Disordered Speech Recognition
interspeech 2024
https://www.isca-archive.org/interspeech_2024/jiang24b_interspeech.pdf?utm_source=chatgpt.com
用一个可训练的 Perceiver 把“可变长度的输入语音”编码成固定长度 speaker prompt，从而能更灵活地适配不同说话人；并报告在中文构音障碍数据上相对 CER 降低（最高 13.04%

Towards Inclusive ASR: Investigating Voice Conversion for Dysarthric Speech Recognition in Low-Resource Languages
interspeech 2025

利用英语构音障碍语音训练VC模型，捕捉构音障碍的声学与韵律特征，并将其迁移至其他语言的健康语音，从而生成非英语的“构音障碍风格”语音。这一策略不仅首次实现了在无目标语言障碍语音数据的情况下生成构音障碍语音，还为构建更具包容性的多语言ASR系统奠定基础引入音高、能量、节奏等多维特征的失调
diffusion/flow 类模型通常更擅长生成复杂分布（病理音质/异常韵律往往就是复杂分布）。
更容易引入强度控制（比如 severity embedding、guidance scale 等），这恰好可以模拟轻度/中度/重度的连续谱。
2025 ACL 的R-VC 就非常直指这个点：提出 rhythm-controllable 的 zero-shot VC（关注时长建模与可控节奏）。
换成更强的多语 SSL 表示/多语离散单元（或者直接走 codec token + semantic token 的混合方案
给 VC 增加 severity embedding（或 guidance scale）
或者在 reference 选择上按严重度分桶采样
或者学一个连续的 style manifold（例如用对比学习把 dysarthria style 拉开）
基于音素级混淆矩阵的扰动（结合 forced alignment）
或引入 articulatory features / phone posteriorgram 的结构性扰动
或把 VC 变成“内容轻微损伤但可控”的生成（这会更贴近真实障碍）
用 中文构音障碍 去训练/微调 VC，再迁移到其他低资源语言或方言；
或者做更细的研究：中文声调损伤、停连/节奏异常 对 ASR 错误类型（插入/删除/替换）的影响，这在英语里不够突出，但在中文里很有研究价值。
这篇论文里也观察到 speaker-only 已经很强，speaker+prosody 通常更强但可能牺牲自然度，这个现象在中文里会更明显（因为 prosody 更“语义敏感”）
中文可以
声调保真的障碍韵律迁移（只注入节奏/能量异常 + F0 抖动/范围压缩，但不破坏四声轮廓）；
用现有中文构音障碍语料（如 CDSD/MDSC）做 VC 微调，再把 AISHELL/WenetSpeech 等健康中文大语料转成障碍风格，训练一个“更可用”的中文 DSR。建立全国地方方言病理风格库，利用CosyVoice跨语言迁移英文病态特征至中文语音，低成本实现数据无限扩充。
https://arxiv.org/abs/2505.14874

Unsupervised Rhythm and Voice Conversion to Improve ASR on Dysarthric Speech
interspeech 2025
引入无监督节奏建模，针对性压缩元音与停顿时长，构建ASR前的“语速正骨”模块，消除拖沓卡顿。  也可以帮助后续的声音重建，把不正常语速变为正常语速
https://arxiv.org/abs/2506.01618

On-the-fly Routing for Zero-shot MoE Speaker Adaptation of Speech Foundation Models for Dysarthric Speech Recognition
interspeech 2025
可以尝试把重度、中度、轻度，或者痉挛型、迟缓型的数据分开，练出 3-5 个不同的 LoRA，尝试训练一个极小的网络（几层 MLP），输出是这几个 LoRA 的权重，自动根据用户特点路由
https://arxiv.org/pdf/2412.18832
，
Dysarthric Speech Recognition Using Curriculum Learning and Articulatory Feature Embedding
interspeech 2024
通过先在健康对照语音上进行词汇层面的预适配，再在患者语音上进行个体化微调，并结合 神经元冻结策略与数据增强，有效缓解了构音障碍语音识别中“说话人差异大、数据稀缺”的难题
https://www.isca-archive.org/interspeech_2024/hsieh24_interspeech.pdf

Enhancing Dysarthric Speech Recognition for Unseen Speakers via Prototype-Based Adaptation
interspeech 2024
DS-TCN 不错，但目前工业界（如小米、华为、Google）在端侧 KWS 上倾向于使用 BC-ResNet 或 MatchboxNet。它们在同样参数量下，抗噪能力和准确率通常优于 TCN 结构。确立冻结解码器、微调编码器的参数策略，结合两阶段迁移，最大化保留通用语言能力
https://arxiv.org/abs/2407.18461

CoLM-DSR: Leveraging Neural Codec Language Modeling for Multi-Modal Dysarthric Speech Reconstruction
InterSpeech 2024
核心借鉴Codec归一化思想，通过解耦音色与韵律，实现S2S潜空间修复
https://www.isca-archive.org/interspeech_2024/chen24t_interspeech.pdf

Exploring Pre-trained Speech Model for Articulatory Feature Extraction in Dysarthric Speech Using ASR
interspeech 2024
核心借鉴预训练模型潜层包含构音运动信息的思想，利用LoRA微调paraformer激活该能力，低成本突破重度识别瓶颈
https://www.isca-archive.org/interspeech_2024/lin24e_interspeech.pdf

Robust Cross-Etiology and Speaker-Independent Dysarthric Speech Recognition
icassp 2025
模型总是会尝试具体记住具体的人， 而不是通用的点，可以加一个遗忘分支和说话人识别loss
https://arxiv.org/html/2501.14994v1

Dysarthric Speech Conformer: Adaptation for Sequence-to-Sequence Dysarthric Speech Recognition
ICASSP 2025 
https://ieeexplore.ieee.org/document/10889046
冻结decoder，微调encoder
数据增强
- SpecAugment：在频率轴和时间轴上随机遮挡或拉伸，模拟语音信号中的多样性；
- 时频扰动：通过改变语速和音高，生成额外的训练样本；
- 语音增强与去噪：对有背景噪声的录音进行清理，生成“干净版”数据，同时保留原始样本，以扩充训练集。
- 损失函数采用 70% KL 散度 + 30% CTC 的

Parameter-efficient Dysarthric Speech Recognition Using Adapter Fusion and Householder Transformation
InterSpeech 2023
论文核心在于利用反射正交矩阵来代替全连接层。这在数学上非常优美，确实能极大幅度压缩参数。如果你在做 极致边缘端（如单片机、超低功耗芯片） 部署，且不支持 LoRA 算子加速时，这个方法依然是“省参数”的神技。
https://arxiv.org/html/2306.07090v1

Zero-Shot Recognition of Dysarthric Speech Using Commercial Automatic Speech Recognition and Multimodal Large Language Models
https://arxiv.org/abs/2512.17474
采用本地Paraformer处理高置信度语音，低置信度样本触发云端GPT-5/Gemini-3。利用原生多模态大模型的能力，结合用户历史发音的In-Context Learning，实现对疑难病理语音的终极纠错。
eess

Raw acoustic-articulatory multimodal dysarthric speech recognition
computer speech & language
https://www.sciencedirect.com/science/article/pii/S0885230825000646

Empowering Dysarthric Communication: Hybrid Transformer-CTC based Speech Recognition System
IEEE Access
[图片]
1)时长变换来模拟患者发音时的停顿和拖长音效应，或者通过语速加快来模拟某些患者在情绪激动时可能出现的语速加速。2)在训练数据中加入背景噪声、语音遮挡等方式，能够增强模型对环境干扰的适应性   3)端到端训练
https://ieeexplore.ieee.org/abstract/document/10993356

Speech Vision: An End-to-End Deep Learning-Based Dysarthric Automatic Speech Recognition System
TNSRE
https://ieeexplore.ieee.org/document/9508383

即使音素崩了，靠整体声学特征也能识别

TWO-STEP ACOUSTIC MODEL ADAPTATION FOR DYSARTHRIC SPEECH  RECOGNITION
ICASSP 2020
尝试基于通用病理，进行个人定制双阶LoRA微调策略，构建从开箱即用到极致定制的体验阶梯。
https://ieeexplore.ieee.org/abstract/document/90537

Self-Supervised ASR Models and Features for Dysarthric and Elderly Speech Recognition
ACM2024 
声学+ 虚拟发音的隐式多模态。不仅微调，更要特征融合。引入虚拟发音特征（A2A）辅助识别，并探索老年认知症筛查的第二增长曲线。
https://ieeexplore.ieee.org/abstract/document/10584335


Speaker adaptation for Wav2vec2 based dysarthric ASR
Interspeech 2022
尝试用LoRA替代Adapter
https://arxiv.org/pdf/2204.00770

Dysarthric Speech Recognition From Raw Waveform with Parametric CNNs Zhengjun Yue, 
InterSpeech 2023
Stream A : 传统的 Fbank -> Paraformer Encoder。保证基础识别率。
Stream B : Raw Waveform -> PCNN (SincNet) -> Downsample -> Linear。
- 用一个轻量级的 SincNet 提取波形特征。
- 将提取出的特征拼接到 Fbank 后面。
保留Fbank主路，旁路引入轻量级SincNet提取波形特征，捕捉被传统特征遗漏的病理高频细节，采用了生物仿生听觉前端，不像其他模型用死板的公式听声音，而是像人耳一样自适应地捕捉患者的特殊频段
https://kclpure.kcl.ac.uk/ws/portalfiles/portal/176300344/INTERSPEECH_2022.pdf


TTS（Dysarthria 语音重建）
Facilitating Personalized TTS for Dysarthric Speakers Using Knowledge Anchoring and Curriculum Learning
interspeech 2025
https://arxiv.org/abs/2508.10412

DiffDSR: Dysarthric Speech Reconstruction Using Latent Diffusion Model
interspeech 2025
demo：https://chenxuey20.github.io/DiffDSR
https://arxiv.org/abs/2506.00350
[图片]

关注点
重建音色（speaker similarity） + 重建内容（wer、word error rate）
- 潜在扩散
  - 参考 NaturalSpeech2
- 说话者身份编码 - VQ 模型 Codec
  - ClearerVoice - 效果不好
  - EnCodec - 很常规的语音 codec
- 语音内容编解码 - SSL 模型
  - Wav2Vec 2.0
  - HuBERT
  - WavLM

数据
- UASpeech
  - 19 位构音障碍者，包含 765 个独立词汇
- VCTK
  - 105 个说话人
- LibriTTS 
  - 2456 名说话者 580 小时

训练
所有内容编码器首先在 LibriSpeech 上训练，每 100 万步，批次为 16，然后在 UASpeech 的目标说话者上进行 2000 步的微调，以提高音素预测的准确性。

[图片]
[图片]


Voice Reconstruction through Large-Scale TTS Models: Comparing Zero-Shot and Fine-tuning Approaches to Personalise TTS in Assistive Communication
interspeech 2025
https://www.isca-archive.org/interspeech_2025/szekely25_interspeech.pdf

Fairness in Dysarthric Speech Synthesis: Understanding Intrinsic Bias in Dysarthric Speech Cloning using F5-TTS
interspeech 2025
https://arxiv.org/abs/2508.05102

❌ CoLM-DSR: Leveraging Neural Codec Language Modeling for Multi-Modal Dysarthric Speech Reconstruction
interspeech 2024
https://arxiv.org/abs/2406.08336
https://arxiv.org/pdf/2406.09873v1
 
弃用离散LM，确立基于Latent Diffusion/Flow Matching的语音修复架构。利用WavLM提取语义骨架，结合课程学习策略微调F5-TTS，在保留患者音色的同时重构清晰韵律，解决Zero-Shot复制病态特征的问题。

EXTENDING PARROTRON: AN END-TO-END, SPEECH CONVERSION AND SPEECH RECOGNITION MODEL FOR ATYPICAL SPEECH
icassp 2021
[图片]
 SpecAugment 数据增强技术作为正则化手段。这种技术的核心逻辑是在模型训练阶段，对输入的语音频谱图进行随机 “遮挡”—— 在频率维度上随机屏蔽部分频段，在时间维度上随机屏蔽部分时长片段
基于定制文本转语音（TTS）的 Bootstrapping 数据增强流程，通过“真实数据适配 - 合成数据生成 - 高质量样本筛选 - 模型再适配” 的闭环，实现数据量与数据质量的双重提升。
转换 - 识别” 一体化且低参数增量；在性能优化上，结合 SpecAugment 正则化与定制 TTS 数据增强，有效破解了非典型语音数据稀缺、模型鲁棒性不足的问题；实验证实，模型仅需少量真实数据适配，就能使 8 种障碍类型语音的平均频谱图 WER 相对降低 76%，重度障碍语音 WER 可降至 20% 左右，显著改善了言语障碍人群的沟通与智能接口使用体验。
https://google.github.io/tacotron/publications/parrotron/
https://arxiv.org/abs/1904.04169


Few-shot dysarthric speech recognition with text-to-speech data augmentation
interspeech 2023
合成语音作为“增量数据”确实能帮 ASR（在已见说话人场景）；
但在 unseen speaker 的 few-shot 场景，仅靠合成语音训练 ASR 还不够用，质量/多样性是瓶颈
https://publications.idiap.ch/attachments/papers/2023/Hermann_INTERSPEECH_2023.pdf



Recent Progress in the CUHK Dysarthric Speech Recognition System in TASLP 2021, [paper]
Journal paper & preprint
Self-Supervised ASR Models and Features for Dysarthric and Elderly Speech Recognition in TASLP 2024, [paper]
Personalized Adversarial Data Augmentation for Dysarthric and Elderly Speech Recognition in TASLP 2024, [paper]
Speaker Adaptation Using Spectro-Temporal Deep Features for Dysarthric and Elderly Speech Recognition in TASLP 2022, [paper]
Detecting Neurocognitive Disorders through Analyses of Topic Evolution and Cross-modal Consistency in Visual-Stimulated Narratives in arxiv 2025, [paper]
Homogeneous Speaker Features for on-the-Fly Dysarthric and Elderly Speaker Adaptation and Speech Recognition, TASLP2025, arxiv
Exploring Cross-Utterance Speech Contexts for Conformer-Transducer Speech Recognition Systems, TASLP25
Conference paper & preprint
2025
Phone-purity Guided Discrete Tokens for Dysarthric Speech Recognition in ICASSP 2025, [paper]
On-the-fly Routing for Zero-shot MoE Speaker Adaptation of Speech Foundation Models for Dysarthric Speech Recognition in INTERSPEECH 2025, paper
MOPSA: Mixture of Prompt-Experts Based Speaker Adaptation for Elderly Speech Recognition, https://arxiv.org/abs/2505.24224
Regularized Federated Learning for Privacy-Preserving Dysarthric and Elderly Speech Recognition, https://arxiv.org/abs/2506.11069
2024
Enhancing Pre-trained ASR System Fine-tuning for Dysarthric Speech Recognition using Adversarial Data Augmentation in ICASSP 2024, [paper]
Exploiting Audio-Visual Features with Pretrained AV-HuBERT for Multi-Modal Dysarthric Speech Reconstruction in ICASSP 2024, [paper]
Perceiver-Prompt: Flexible Speaker Adaptation in Whisper for Chinese Disordered Speech Recognition in InterSpeech 2024, [paper]
Towards Automatic Data Augmentation for Disordered Speech Recognition in ICASSP 2024, [paper]
Not All Errors Are Equal: Investigation of Speech Recognition Errors in Alzheimer’s Disease Detection in ISCSLP 2024, [paper]
Structured Speaker-Deficiency Adaptation of Foundation Models for Dysarthric and Elderly Speech Recognition in arxiv 2024, [paper]
Devising a Set of Compact and Explainable Spoken Language Feature for Screening Alzheimer’s Disease in ISCSLP 2024, [paper]
Towards Within-Class Variation in Alzheimer’s Disease Detection from Spontaneous Speech in arxiv 2024, [paper]
2023
Adversarial Data Augmentation Using VAE-GAN for Disordered Speech Recognition in ICASSP 2023, [paper]
Exploiting prompt learning with pre-trained language models for Alzheimer's Disease detection in ICASSP 2023, [paper]
Exploring Self-supervised Pre-trained ASR Models For Dysarthric and Elderly Speech Recognition in ICASSP 2023, [paper]
On-the-Fly Feature Based Rapid Speaker Adaptation for Dysarthric and Elderly Speech Recognition in InterSpeech 2023, [paper]
Use of Speech Impairment Severity for Dysarthric Speech Recognition in InterSpeech 2023, [paper]
Exploiting Cross-domain And Cross-Lingual Ultrasound Tongue Imaging Features For Elderly And Dysarthric Speech Recognition in InterSpeech 2023, [paper]
Leveraging Pretrained Representations with Task-related Keywords for Alzheimer's Disease Detection in ICASSP 2023, [paper]
Hyper-parameter Adaptation of Conformer ASR Systems for Elderly and Dysarthric Speech Recognition in InterSpeech 2023, [paper]
Integrated and Enhanced Pipeline System to Support Spoken Language Analytics for Screening Neurocognitive Disorders in InterSpeech 2023, [paper]
2022 and before
Exploiting Cross Domain Acoustic-to-articulatory Inverted Features For Disordered Speech Recognition, https://arxiv.org/abs/2203.10274
Speaker Identity Preservation in Dysarthric Speech Reconstruction by Adversarial Speaker Adaptation, https://arxiv.org/abs/2202.09082
VCVTS: Multi-speaker Video-to-Speech synthesis via cross-modal knowledge transfer from voice conversion, https://arxiv.org/abs/2202.09081
Confidence Score Based Conformer Speaker Adaptation for Speech Recognition, https://arxiv.org/abs/2206.12045
Bayesian Parametric and Architectural Domain Adaptation of LF-MMI Trained TDNNs for Elderly and Dysarthric Speech Recognition, https://www.isca-archive.org/interspeech_2021/deng21d_interspeech.html
Exploiting Cross-Domain Visual Feature Generation for Disordered Speech Recognition, paper