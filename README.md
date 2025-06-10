# Leveraging LLMs as an Augmentation to Traditional Hyperparameter Tuning

A demo that leverages Large Language Models to automatically analyze, design, and refine neural network architectures during the training process by monitoring gradient flows. The associated blog post can be found [here](https://aws.amazon.com/blogs/hpc/leveraging-llms-as-an-augmentation-to-traditional-hyperparameter-tuning/).

## Overview
This project demonstrates how LLMs can serve as an alternative to traditional hyperparameter tuning, using Claude (via AWS Bedrock) to analyze training metrics, gradient flows, and model stability. The system can recommend and implement both architecture changes and hyperparameter adjustments to improve model performance without requiring exhaustive parameter searches.

As described in the accompanying blog post, this approach leverages the collective knowledge embedded in LLMs to act as "surrogate universal experts" for neural architecture design.

## Key Features

* **Gradient-Based Architecture Analysis:** Uses gradient norms as diagnostic signals to identify architectural issues
* **LLM-Guided Model Evolution:** Automatically modifies architectures based on training dynamics
* **Agentic Workflow with LangGraph:** Orchestrates the training process with decision-making around when to modify the model
* **Multi-Agent System:** Uses two LLMs in collaboration - one for reasoning about architecture changes and one for testing/fixing code
* **Resilient API Interactions:** Implements exponential backoff to handle API throttling
* **Comprehensive Evaluation:** Tracks detailed metrics on model performance and stability
* **MultiGPU Training Support:** Works across multiple GPUs using PyTorch DDP


## Files in this Repository

* **cnn_model.py:** Initial baseline CNN implementation (intentionally flawed)
* **LLM_designers.py:** Main script implementing the LLM-powered model optimization workflow
* **ONLY_train.py:** Simplified training script without LLM assistance (for benchmarking)

## Requirements
* Python 3.8+
* PyTorch 2.0+
* AWS account with Bedrock access
* LangGraph
* Tenacity for exponential backoff
* tqdm, matplotlib, numpy
* AWS Boto3

## Installation

### Clone the repository

```
git clone git@github.com:aws-samples/sample-Leveraging-LLMs-Augmentation-Traditional-Hyperparameter-Tuning.git
cd sample-Leveraging-LLMs-Augmentation-Traditional-Hyperparameter-Tuning
```

### Install required packages

```
pip install torch torchvision langchain langgraph boto3 matplotlib tqdm tenacity
```

### AWS Configuration

1. Configure your AWS credentials:
```
aws configure
```
2. Ensure you have access to the Bedrock service and Claude models.

### Usage

**Traditional Training (Without LLM Optimization)**
```
python ONLY_train.py 
```

**Traditional Training (With LLM Optimization)**
```
python ONLY_train.py --use_LLM_config
```

**LLM-Guided Optimization**

There are a few command line options a user can modify, but we rather users test this out in an iPython window to understand how the code works. At the top of this file is a global variable ```CROSS_REGION_ARN```.  A user will need to use the cross region ARN from their account. Navigate to Bedrock in the AWS console -> Scroll down on the left side and click "Cross-region inference" -> Copy the "Inference profile ARN" in the python variable. For the blog post we used Claude 3.7 Sonnet. Try different models to see how well they do!

```
python LLM_designers.py 
```

**Key Arguments**
* --batch_size: Batch size for training
* --learning_rate: Initial learning rate
* --epochs: Total training epochs
* --apply_changes: Whether to apply LLM-recommended changes to the model
* --analyze_interval: How many epochs to wait before asking the LLM for analysis
* --aws_region: AWS region for Bedrock API calls
* --claude_model_id: Claude model to use for optimization

## How It Works

1. **Initial Training:** The system begins training a basic CNN model while tracking gradient norms and other metrics
2. **Periodic Analysis:** At specified intervals (e.g., every 50 epochs), the LLM analyzes training performance
3. **Decision Routing:** Based on stability assessment, the system either continues training, modifies architecture, or adjusts hyperparameters
4. **Architecture Generation:** When needed, a new architecture is generated that addresses identified issues
5. **Code Validation:** A second LLM tests and fixes any issues in the generated code
6. **Iterative Improvement:** The process continues until the model is deemed stable or training completes

## Workflow Diagram
The repository implements a multi-agent workflow as described in the blog post, with components for:

* Training and gradient monitoring
* Model health analysis
* Architecture modification
* Hyperparameter tuning
* Error handling and code fixing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.
