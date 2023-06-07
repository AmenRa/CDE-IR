# CDD for IR Experiments

Disclaimer: This is a work in progress.

## What is this repo?
This repo is a boilerplate for applying Configuration-Driven Development (CDD) to Information Retrieval (IR) experiments.
It supports experiments on [MSMARCO Passage Ranking v1](https://microsoft.github.io/msmarco) and implements the following Neural Retrieval architectures: BiEncoder, CrossEncoder, ColBERT.

## What is CDD?
Configuration-Driven Development (CDD) is a software development approach that emphasizes the use of configuration files to define and control the behavior of an application.
In CDD, instead of hard-coding specific values or logic directly into the source code, developers rely on external configuration files to specify various aspects of the application's behavior.

## Why CDD for Information Retrieval experiments?

CDD can be particularly useful for IR experiments for several reasons:

1. **Flexibility in Experiment Setup**: IR experiments often involve testing different retrieval models, algorithms, parameters, or data preprocessing techniques. CDD allows researchers to define and modify these experimental configurations without changing the source code directly. This flexibility enables quick iteration and experimentation with various settings.

2. **Reproducibility**: Reproducibility is crucial in IR research to validate and compare different approaches. Using configuration files to define the experimental setup, researchers can precisely document the specific configuration used for a particular experiment, making it easier for others to replicate and validate results.

3. **Modularity and Maintainability**: IR experiments often involve multiple components, such as indexing, query processing, relevance ranking, and evaluation metrics. CDD allows researchers to modularize these components and configure them separately. Each module can have its own configuration file, making it easier to maintain, update, and reuse components across different experiments.

4. **Customization for Different Scenarios**: IR systems may need to be adapted to different domains, data collections, or evaluation scenarios. With CDD, researchers can easily customize the configuration files to adjust the system's behavior and parameters according to specific requirements. This flexibility allows researchers to evaluate and compare different configurations under various scenarios.

5. **Collaboration and Sharing**: Configuration files can serve as a common language between researchers, facilitating the collaboration and sharing of experimental setups. Researchers can share their configuration files with others, enabling replication, extension, or modification of experiments while promoting knowledge sharing and fostering a more collaborative research environment.

By adopting Configuration-Driven Development in IR experiments, researchers can streamline the experimentation process, enhance reproducibility, promote collaboration, and facilitate the customization and adaptation of retrieval systems for different scenarios, leading to more robust and reliable research outcomes.

## Software stack
- [PyTorch](https://pytorch.org)
- [PyTorch Lightning](https://www.pytorchlightning.ai/index.html)
- [ir-datasets](https://ir-datasets.com)
- [ranx](https://github.com/AmenRa/ranx)