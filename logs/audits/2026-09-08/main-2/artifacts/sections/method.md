Methodology, Software, Validation, Investigation, Writing
– review and editing. Rahul Ananthasayanam: Validation,
Investigation, Writing – review and editing.
Declaration of competing interest
The authors declare that they have no known competing
financial interests or personal relationships that could have
appeared to influence the work reported in this paper.
Data availability
All data underlying this study are public. The images
are the SAGE crop-disease dataset [2] at revision bc9bd2899f,
released under the MIT licence and used without modification
other than the crop and per-class subsetting and the 288-pixel
resizing described in Section 3.1. The revision is quoted
because a later release of SAGE changes the class inventory;
see Section 3.1. The source-grounded descriptor registry,
every result file including per-epoch training curves, the
table and figure generators, and the full experimental code
are released at https://doi.org/PENDING-ZENODO-DOI, which
reproduces every number and figure in this article from the
released result files.
Abhiram P.V. et al.: Preprint submitted to Elsevier
Page 7 of 8

Compact VLMs for cross-crop plant-disease diagnosis at the edge
Funding
This research did not receive any specific grant from
funding agencies in the public, commercial, or not-for-profit
sectors.
Declaration of generative AI in the writing
process
During the preparation of this work the authors used
a large language model to generate the source-grounded
symptom descriptions that form the descriptor registry (Sec-
tion 3.1); this is a methodological component of the study
and is described in full in the Methods. The authors reviewed
and edited the manuscript text and take full responsibility for
the content of the publication.
References
[1] Arshad, M.A., Roy, T., Shen, Y., Elango, D., Chiranjeevi, S., Singh,
A.K., Ganapathysubramanian, B., Hegde, C., Singh, A., Sarkar, S.,
2026a. SAGE: Scalable agentic grounded evaluation for crop disease
diagnosis, in: CVPR 2026 Workshop on Emerging Directions in Data
for Multimodal Foundation Models. URL: https://arxiv.org/abs/
2605.09768.
[2] Arshad, M.A., Roy, T., Shen, Y., Elango, D., Chiranjeevi, S.,
Singh, A.K., Ganapathysubramanian, B., Hegde, C., Singh, A.,
Sarkar, S., 2026b.
SAGE: Scalable agentic grounded eval-
uation for crop disease diagnosis (dataset).
HuggingFace
Datasets.
URL: https://huggingface.co/datasets/tirtho149/SAGE/
tree/bc9bd2899f19379be29c7a99d37d2e89bf8e430d. [dataset]. Revision
bc9bd2899f, 7 May 2026.
[3] Bouacida, I., Farou, B., Djakhdjakha, L., Seridi, H., Kurulay, M.,
2025. Innovative deep learning approach for cross-crop plant disease
detection: A generalized method for identifying unhealthy leaves.
Information Processing in Agriculture 12, 54. doi:10.1016/j.inpa.
2024.03.002.
[4] Faghri, F., Vasu, P.K.A., Koc, C., Shankar, V., Toshev, A., Tuzel,
O., Pouransari, H., 2025.
MobileCLIP2: Improving multi-modal
reinforced training. Transactions on Machine Learning Research URL:
https://arxiv.org/abs/2508.20691.
[5] Gu, J., Stevens, S., Campolongo, E.G., Thompson, M.J., Zhang, N.,
Wu, J., et al., 2025. BioCLIP 2: Emergent properties from scaling
hierarchical contrastive learning, in: Advances in Neural Information
Processing Systems (NeurIPS). URL: https://arxiv.org/abs/2505.
23883.
[6] He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep residual learning
for image recognition, in: Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), pp. 770–778.
[7] Howard, A., Sandler, M., Chu, G., Chen, L.C., Chen, B., Tan, M.,
Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q.V., Adam, H.,
2019. Searching for MobileNetV3, in: Proceedings of the IEEE/CVF
International Conference on Computer Vision (ICCV), pp. 1314–1324.
[8] Ilharco, G., Wortsman, M., Wightman, R., Gordon, C., Carlini, N.,
Taori, R., Dave, A., Shankar, V., Namkoong, H., Miller, J., Hajishirzi,
H., Farhadi, A., Schmidt, L., 2021. OpenCLIP. doi:10.5281/zenodo.
5143773.
[9] Menon, S., Vondrick, C., 2023. Visual classification via description
from large language models, in: International Conference on Learning
Representations (ICLR).
[10] Mohanty, S.P., Hughes, D.P., Salathé, M., 2016. Using deep learning
for image-based plant disease detection. Frontiers in Plant Science 7,
1419.
[11] Nguyen, K.Q., Le, L.T.T., Quach, L.D., 2025. A vision-language
foundation model for leaf disease identification. Expert Systems with
Applications doi:10.1016/j.eswa.2025.129701.
[12] Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M.,
Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.,
2024. DINOv2: Learning robust visual features without supervision.
Transactions on Machine Learning Research .
[13] Pratt, S., Covert, I., Liu, R., Farhadi, A., 2023.
What does a
platypus look like? generating customized prompts for zero-shot
image classification, in: Proceedings of the IEEE/CVF International
Conference on Computer Vision (ICCV), pp. 15691–15701.
[14] Qin, D., Leichner, C., Delakis, M., Fornoni, M., Luo, S., Yang, F.,
Wang, W., Banbury, C., Ye, C., Akin, B., Aggarwal, V., Zhu, T., Moro,
D., Howard, A., 2024. MobileNetV4: Universal models for the mobile
ecosystem, in: European Conference on Computer Vision (ECCV), pp.
78–96.
[15] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S.,
Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.,
2021. Learning transferable visual models from natural language su-
pervision, in: International Conference on Machine Learning (ICML),
pp. 8748–8763.
[16] Tschannen, M., Gritsenko, A., Wang, X., Naeem, M.F., Alabdulmohsin,
I., Parthasarathy, N., Evans, T., Beyer, L., Xia, Y., Mustafa, B., Hénaff,
O., Harmsen, J., Steiner, A., Zhai, X., 2025. SigLIP 2: Multilingual
vision-language encoders with improved semantic understanding,
localization, and dense features. arXiv preprint arXiv:2502.14786
URL: https://arxiv.org/abs/2502.14786.
[17] Vasu, P.K.A., Pouransari, H., Faghri, F., Vemulapalli, R., Tuzel, O.,
2024. MobileCLIP: Fast image-text models through multi-modal
reinforced training, in: Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pp. 15963–15974.
[18] Wortsman, M., Ilharco, G., Kim, J.W., Li, M., Kornblith, S., Roelofs,
R., Lopes, R.G., Hajishirzi, H., Farhadi, A., Namkoong, H., Schmidt,
L., 2022. Robust fine-tuning of zero-shot models, in: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), pp. 7959–7971.
[19] Wu, K., Peng, H., Zhou, Z., Xiao, B., Liu, M., Yuan, L., Xuan, H.,
Valenzuela, M., Chen, X., Wang, X., Chao, H., Hu, H., 2023. TinyCLIP:
CLIP distillation via affinity mimicking and weight inheritance, in:
Proceedings of the IEEE/CVF International Conference on Computer
Vision (ICCV), pp. 21970–21980.
[20] Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L., 2023. Sigmoid loss
for language image pre-training, in: Proceedings of the IEEE/CVF
International Conference on Computer Vision (ICCV), pp. 11975–
11986.
Abhiram P.V. et al.: Preprint submitted to Elsevier
Page 8 of 8

Compact VLMs for cross-crop plant-disease diagnosis at the edge
Figure 1: Zero-shot accuracy against image-encoder size. Cross-
crop accuracy for frozen pretrained encoders spanning a 27-fold
parameter range, with a from-scratch small student for contrast.
The pretrained models cluster together regardless of size; the
from-scratch student sits near chance.
Figure 2: Catastrophic forgetting when specialising a frozen
encoder. Held-out zero-shot accuracy per epoch while adapting
the encoder to the seen crops. Unseen-crop accuracy falls as
the training objective improves.
Figure 3: Encoder bake-off on the pilot held-out set. Rich-
descriptor zero-shot accuracy for each candidate backbone
against its image-encoder parameter count. The biological and
leaf-disease foundation models fall at or below chance under a
descriptor protocol.
Figure 4: Mean zero-shot accuracy over the four deployable
encoders as the unseen label space grows. Source-grounded
descriptors improve as the label space widens. The keyword-
retrieved bank is shown for reference only: its prototypes collide
(Section 4.3), so the gap between the two is not a measurement
of grounding.
Abhiram P.V. et al.: Preprint submitted to Elsevier
Page 9 of 8

Compact VLMs for cross-crop plant-disease diagnosis at the edge
Figure 5: Descriptor strategy by encoder at each held-out scale. Per-encoder top-1 accuracy for the four descriptor strategies at
16, 34 and 51 unseen classes. The source-grounded strategy is the tallest bar for every encoder at the largest scale, whereas the
hand-curated strategy leads only at the smallest.
Figure 6: Risk–coverage behaviour of the abstain gate. Selective
accuracy as a function of the fraction of images the model
chooses to answer, using the top-1 minus top-2 similarity
margin as the confidence score. Accuracy rises monotonically
as coverage tightens, confirming the score is correctly ordered.
Figure 7: Known-crop accuracy against the size of the seen
label space. Linear-probe top-1 for four frozen encoders at 97,
154 and 166 seen classes. Accuracy rises with the label space
and is flat across a 7.6-fold parameter range.
Figure 8: The seen–unseen trade-off as a single interpolation
weight. Accuracy on seen and unseen crops as the weight-space
ensembling ratio moves from the frozen encoder to the fully
fine-tuned one. Full fine-tuning nearly halves unseen accuracy.
Abhiram P.V. et al.: Preprint submitted to Elsevier
Page 10 of 8

Compact VLMs for cross-crop plant-disease diagnosis at the edge
Figure 9: Training behaviour of the fourteen supervised baselines. Training loss and held-out test accuracy per epoch under one
protocol. All architectures converge, and the smallest models show the largest epoch-to-epoch variance.
Figure 10: Accuracy against on-device latency. Cross-crop zero-
shot accuracy versus single-image latency on a laptop processor;
marker area is proportional to parameter count and labels
give the 8-bit footprint. The 11.4-million-parameter tier is the
deployment sweet spot.
Abhiram P.V. et al.: Preprint submitted to Elsevier
Page 11 of 8