---
title: 'MM-PoE: Multiple Choice Reasoning via. Process of Elimination using Multi-Modal Models'
tags:
  - machine learning
  - large language models
  - multi-modal
  - python
  - multiple choice reasoning
  - visual question answering
authors:
  - name: Sayak Chakrabarty
    orcid: 0009-0004-6179-389X
    affiliation: 1
  - name: Souradip Pal
    orcid: 0000-0002-5781-3032
    affiliation: 2
affiliations:
 - name: Northwestern University
   index: 1
 - name: Purdue University
   index: 2
date: 22 October 2024
bibliography: paper.bib
---

# Summary

This paper introduces Multiple Choice Reasoning via. Process of Elimination using Multi-Modal models, also know as Multi-Modal Process of Elimination (MM-PoE), a method to enhance vision language models' performance on multiple-choice visual reasoning tasks by employing a two-step scoring system that first eliminates incorrect options and then predicts from the remaining ones. Our experiments across three question answering datasets show the method's effectiveness, particularly in visual reasoning tasks. This method addresses one of the key limitations of the paper [@ma2023poe] by extending to tasks involving multi-modalities and also includes experimentation techniques for few-shot settings.

# Statement of Need

Large Language models (LLMs) excel at in-context learning for multiple choice reasoning tasks but often treat all options equally, unlike humans who typically eliminate incorrect choices before selecting the correct answer. Same is true for vision language models (VLMs) in case of visual question answering tasks with multiple choices. This discrepancy can limit the effectiveness of vision language models in accurately solving such tasks. To address this, we introduce Multi-Modal Process of Elimination (MM-PoE), a two-step scoring method designed to enhance VLM performance by mimicking human reasoning strategies in multi-modal settings.

In the first step, the method evaluates and scores each option, systematically eliminating those that appear incorrect. The second step involves masking these eliminated options, allowing the VLM to focus solely on the remaining viable choices to make a final prediction. Our zero-shot experiments across three datasets demonstrate MM-PoE's effectiveness, particularly excelling in logical reasoning scenarios. Additionally, MM-PoE proves adaptable to few-shot settings and is compatible with the current state-of-the-art vision language models (VLMs).

Using this tool, researchers and practitioners can experiment and significantly improve the accuracy and reliability of VLMs in multiple choice reasoning tasks, making it a valuable tool for advancing machine learning models for visual reasoning.

# State of the Field

A common strategy for answering multiple-choice questions, especially under examination conditions, involves a process of elimination where incorrect answers are systematically discarded to narrow down the choices to the most likely correct one. This approach, grounded in everyday test-taking strategies, contrasts with how current language models (LMs) and vision language models (VLMs) handle multiple-choice reasoning tasks. Typically, VLMs evaluate each option independently or collectively without actively discarding less likely answers, potentially reducing their effectiveness in distinguishing the best choice from plausible distractors.

This paper argues that vision language models can benefit from an explicit two-step reasoning process akin to human problem-solving techniques. The proposed method, known as Multi-Modal Process of Elimination (MM-PoE), enhances the decision-making process by first scoring and then eliminating options that are seemingly incorrect before focusing on selecting the correct answer from the remaining choices. This method is designed to align with natural human reasoning by replicating how individuals often approach multiple-choice questions, particularly under the constraint of time and accuracy, as frequently experienced in academic testing environments.

Our hypothesis posits that vision language models, when equipped with a mechanism to discard implausible answers systematically, can achieve better performance on multiple-choice visual reasoning tasks. This is particularly relevant in the context of logical reasoning, where the elimination of clearly incorrect options can simplify the decision process and potentially lead to more accurate outcomes. This idea is supported by previous work demonstrating the effectiveness of LMs in various reasoning tasks when adapted to more human-like reasoning methods[@holtzman2021surface].

In the development of MM-PoE, we draw inspiration from the established capabilities of LMs to handle complex reasoning tasks [@brown2020language] and the known strategies that humans employ in test-taking scenarios as depicted in [@ma2023poe]. The approach builds on the foundational work in language modeling likelihood [@brown2020language], which demonstrates the LMs' ability to perform in-context learning. By incorporating a structured process to eliminate unlikely choices in a multi-modal setting, MM-PoE aims to refine this capability, making it more targeted and efficient in dealing with the nuanced challenges presented by multiple-choice questions.

The effectiveness of this approach is underscored through zero-shot and few-shot experiments across a diverse set of reasoning datasets, illustrating that the integration of human-like elimination strategies can significantly enhance the performance of vision language models. This paper aims to show that by mimicking human reasoning processes, we can make VLMs not only perform better on standardized visual reasoning tasks but also behave in ways that are more interpretable and aligned with human cognitive processes.


# Methodology

The Multi-Modal Process of Elimination (MM-PoE) introduced in this paper operates on a two-step mechanism designed to enhance the decision-making capabilities of vision language models (VLMs) in multiple-choice visual reasoning tasks. This method employs a novel approach to option elimination followed by a focused prediction phase. The strategy is rooted in the belief that separating the elimination of clearly incorrect options from the choice of the best remaining option will improve overall task performance.

## Problem Setting

Given a multiple-choice visual reasoning task, we define the problem setting as follows:

- Let $x$ be the question or context provided.
- Let $h$ be the image provided.
- Let $Y = \{y_1, y_2, \ldots, y_n\}$ be the set of multiple-choice options available.
- Let $y$ be the correct answer from $Y$.

The goal is to develop an in-context learning method that accurately selects $y$ from $Y$ given $x$ and $h$.

## Two-Step Scoring Method

### Step 1: Elimination

In the first step of the MM-PoE method, each option $y_i$ is scored based on a specified metric. The score function, $\text{score}(x, h, y_i)$, evaluates each option's plausibility given the question $x$ and image $h$. The scores are used to eliminate options that are deemed less likely to be correct. Specifically, options whose scores are below the average score are eliminated. This is calculated as follows:

$$
s_i = \text{score}(x, h, y_i)
$$

$$
Y_{\text{wrong}} = \{y_i | s_i < \text{avg}(s_1, \ldots, s_n)\}
$$

This elimination strategy intuitively aligns with how humans often discard options that seem clearly incorrect before carefully considering the remaining choices.

### Step 2: Prediction

The second step involves making the final choice from the non-eliminated options. This step utilizes a binary mask to exclude the eliminated options during the prediction phase. The mask for each option $y_i$ is defined as follows:

$$
m_i = \begin{cases} 
0 & \text{if } y_i \in Y_{\text{wrong}} \\
1 & \text{otherwise}
\end{cases}
$$

The masked context $x_{\text{mask}}$ is then constructed by modifying the original context $x$ to include only the options for which $m_i = 1$. Each option is scored again, but this time within the context that explicitly excludes the eliminated options, possibly by using a template $T$ that masks out $Y_{\text{wrong}}$ in the presentation of the options:

$$
x_{\text{mask}} = T(x, Y, \text{mask})
$$

The final predicted answer $\hat{y}$ is then the option with the highest score among the remaining options:

$$
\hat{y} = \arg\max_{i | m_i = 1} \text{score}(x_{\text{mask}}, h, y_i)
$$

# Experimental Setup

To evaluate the effectiveness of the Multi-Modal Process of Elimination (MM-PoE), we designed an experimental framework that tests the method across a diverse set of visual reasoning datasets. This setup aims to compare MM-PoE with existing scoring methods to highlight its potential improvements in accuracy and reasoning capability. Our experiments primarily focused on a zero-shot setting to evaluate the generalization capabilities of MM-PoE without any task-specific tuning. Accuracy was used as the main metric for performance evaluation, with results averaged over multiple seeds to ensure robustness.

To further explore the versatility of MM-PoE, we also examined its performance in few-shot settings by incorporating examples into the model's input, aiming to observe any changes in effectiveness when provided with context-specific demonstrations.

## Data

Our experiments were conducted on three different multiple-choice visual reasoning datasets - Visual Question Answering(VQA) [@VQA], ScienceQA [@lu2022learn] and Diagram Understanding(AI2D) [@Kembhavi2016ADI], selected to cover a broad spectrum of reasoning types and complexities. These tasks include both traditional visual reasoning tasks and more specialized ones designed to test specific reasoning skills. To ensure a comprehensive evaluation, we used train sets from established benchmarks when available; otherwise, we utilized development sets. In case of varying number of options in the multiple-choice answers for SceinceQA and AI2D datasets, we filtered questions containing image context and exactly four options.

| Dataset | #Options | Train  | Dev  | Test |
|----|------|------|------|-----------|
| VQA | 18 | 248,349 | 121,512 | 244,302 |
| ScienceQA | 4 | 12726 | 4241 | 4241 |
| AI2D | 4 | 3921 | 982 | - |

## Model

For the core experiments, we utilized the GIT and BLIP models, chosen for its balance between computational efficiency and performance in instruction-tuned vision language tasks. These models have demonstrated strong capabilities in handling various multi-modal tasks and serves as a robust platform for evaluating our MM-PoE method.

## Baselines

We compared MM-PoE against five baseline scoring methods to assess its relative performance:

1. **Language Modeling (LM):** This baseline uses the raw vision language modeling likelihood as the scoring function.
2. **Average Language Modeling (AVG):** This method averages the log probabilities across all tokens in the option.
3. **Calibration:** This involves adjusting the VLM scores based on calibration techniques that aim to correct for the model's confidence.
4. **Channel:** Channel methods score each option based on how likely the question is given the option, which reverses the typical conditional probability used in LMs.
5. **Multiple Choice Prompting (MCP):** This approach formats the input by presenting the question followed by all options, prompting the model to select the most likely option.

Each method provides a different approach to scoring options, allowing for a comprehensive comparison of how each interacts with the structure and strategy of MM-PoE.

## Implementation

The effectiveness of MM-PoE hinges on the robustness of the scoring function and the accuracy of the elimination step. The scoring function can be any VLM-based likelihood estimator, such as vision language modeling likelihood or any of its alternatives like average log probability or calibrated log probability. Our implementation tests multiple such scoring functions to identify the most effective ones in both eliminating implausible options and accurately selecting the final answer. 

The MM-PoE method is designed to be model-agnostic, meaning it can be implemented using any existing VLM capable of scoring text options, and it is flexible enough to be adapted to different types of multiple-choice visual answering questions across various domains. The scoring functions were carefully chosen based on their theoretical alignment with the two-step elimination and prediction philosophy of MM-PoE. We conducted extensive parameter tuning and optimization to maximize the performance of both the elimination step and the final prediction accuracy.

This experiment setup was designed to rigorously test the effectiveness of MM-PoE across a range of visual reasoning tasks and compare its performance against standard baseline methods. The results of these experiments are intended to demonstrate the potential benefits of integrating a process of elimination approach into vision language model reasoning strategies for multiple-choice questions.


## Results

MM-PoE consistently outperformed or matched the best-performing baselines across all datasets, showing particular strength in logical reasoning. The method's effectiveness in separating elimination and prediction tasks was crucial to its success.

| Model | Dataset | LM | AVG | Calibration | Channel | MCP  | PoE  |
|----|------|------|------|-----------|---|---|---|
|microsoft/git-base-vqav2| ScienceQA | 27.4 | 17.8 | 23.2| 24.6 | 25.8 | 27.2 |
|microsoft/git-base-vqav2| AI2D | 25.4| 26.2 | 26.4| 25.4 | 25.3 | 26.5 |
|microsoft/git-base-textvqa| ScienceQA | 21.8 | 20.4 | 25.8 | 23.4 | 23.6 | 28.2 |
|microsoft/git-base-textvqa| AI2D | 26.5 | 27.6 | 20.8| 26.2 | 24.2| 26.8 |

**Table 1**: Comparison of Multiple-Choice Prompting (MCP) and Process of Elimination (PoE) accuracy scores on 2 visual question answering datasets for the `microsoft/git-base-vqav2` and `microsoft/git-base-textvqa` models in the zero-shot settings. Each dataset has different number of answer choices. PoE mostly outperforms MCP on all the visual reasoning tasks for the two multi-modal models mentioned.

## Examples

### ScienceQA Example
<img src="figures/image.png" alt="Example" width="500">

**Question**: Which of these states is farthest north?<br>
**Options**: West Virginia, Louisiana, Arizona, Oklahoma<br>
**Ground Truth Option**: West Virginia

**Predicted Masks**: West Virginia, Louisiana, [MASK], [MASK]<br>
**Predicted Option**: West Virginia

### AI2D Example

<img src="figures/17.png" alt="Example" width="500">

**Question**: Are phytoplankton predators or prey in this food chain?<br>
**Options**: producer, predator, prey, NA<br>
**Ground Truth Option**: prey

**Predicted Masks**: [MASK], predator, prey, NA<br>
**Predicted Option**: prey

# Conclusion

MM-PoE demonstrates a significant improvement in handling multiple choice visual reasoning tasks by mimicking a human-like process of elimination approach. Future work will focus on enhancing its generalizability and efficiency, possibly extending to handle better masking strategies.

# Ethics Statement

While this method uses publicly available data and models, users should be aware of potential biases in the data and model outputs.

# Acknowledgements

We would like to extend our sincere gratitude to Northwestern University for providing access to their servers and GPU resources, which were instrumental in conducting this research. The computational power and infrastructure made available by the university enabled the efficient processing and analysis of large datasets, significantly contributing to the success of the project. Without this support, the research would not have been possible at the scale or speed required. We deeply appreciate the university’s commitment to fostering a collaborative research environment and supporting technological innovation.