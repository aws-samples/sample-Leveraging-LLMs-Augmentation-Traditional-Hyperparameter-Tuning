#!/usr/bin/env python3
# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                #
######################################################################
"""
Created on Mon Mar 24 19:51:07 2025

@author: R. Pivovar

"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import boto3
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Dict, Any, Optional, Tuple, Union
import re
import copy
import logging
import uuid
import shutil
import sys
import importlib
import importlib.util
import tenacity
from botocore.exceptions import ClientError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


CROSS_REGION_ARN = '<user supplied>'


def call_bedrock_with_tenacity(bedrock_client, model_id, prompt, max_tokens=4000, thinking=None):
    """
    Call Bedrock API with tenacity-based exponential backoff for handling throttling issues.

    Args:
        bedrock_client: The Bedrock client instance
        model_id: The model ID to use
        prompt: The prompt to send
        max_tokens: Maximum tokens in the response
        thinking: Dictionary with thinking parameters or None

    Returns:
        The parsed response from the Bedrock API
    """
    # Define the request body based on whether thinking is enabled
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }

    # Add thinking parameters if provided
    if thinking:
        body["thinking"] = thinking

    # Create a function that will be retried
    @tenacity.retry(
        # Retry up to 5 times
        stop=tenacity.stop_after_attempt(5),

        # Exponential backoff with jitter
        wait=tenacity.wait_exponential(multiplier=1, min=1, max=60),

        # Retry on specific exceptions
        retry=(
            tenacity.retry_if_exception_type(ClientError) |
            tenacity.retry_if_exception_message(match="ThrottlingException") |
            tenacity.retry_if_exception_message(match="TooManyRequestsException") |
            tenacity.retry_if_exception_message(match="ServiceUnavailableException") |
            tenacity.retry_if_exception_message(match="InternalServerException")
        ),

        # Before retry callback - for logging
        before_sleep=tenacity.before_sleep_log(logger, logging.INFO),

        # After retry callback - also for logging
        after=tenacity.after_log(logger, logging.INFO)
    )
    def make_api_call():
        try:
            response = bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(body)
            )
            return json.loads(response['body'].read())
        except Exception as e:
            logger.warning(f"Bedrock API call failed: {e.__class__.__name__}: {str(e)}")
            raise

    # Execute the retryable function
    return make_api_call()


def ensure_empty_log_directory(log_dir="./logs"):
    """
    Ensures the specified log directory is empty by removing all files in it.
    If the directory doesn't exist, it creates it.

    Args:
        log_dir (str): Path to the log directory. Defaults to "log".

    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    try:
        # Check if directory exists
        if os.path.exists(log_dir):
            # If it exists, check if it's a directory
            if os.path.isdir(log_dir):
                # Option 1: Remove all files individually
                for filename in os.listdir(log_dir):
                    file_path = os.path.join(log_dir, filename)
                    if os.path.isfile(file_path):
                        try:
                            os.unlink(file_path)
                        except Exception as e:
                            print(f"{e}")
                            continue
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                print(f"All files removed from {log_dir} directory.")

            else:
                # If it's not a directory but a file, remove it and create directory
                os.unlink(log_dir)
                os.makedirs(log_dir)
                print(f"Removed file named {log_dir} and created directory instead.")
        else:
            # If it doesn't exist, create it
            os.makedirs(log_dir)
            print(f"Created new {log_dir} directory.")

        return True
    except Exception as e:
        print(f"Error ensuring empty log directory: {e}")
        return False

def write_changes_summary(recommendations_history, filename="model_changes_summary.md"):
    """Write a readable summary of all changes made to the model"""
    with open(filename, 'w') as f:
        f.write("# Neural Network Evolution History\n\n")
        f.write("This document tracks all changes made to the neural network architecture and hyperparameters.\n\n")

        # Group changes by epoch and type
        arch_changes = [r for r in recommendations_history if r['type'] == 'architecture']
        hp_changes = [r for r in recommendations_history if r['type'] == 'hyperparameters']

        # Write architecture changes
        if arch_changes:
            f.write("## Architecture Changes\n\n")
            for change in arch_changes:
                #print(change)
                f.write(f"**Epoch {change['epoch']}**\n\n")
                f.write(f"Explanation: {change['recommendations']['explanation']}\n\n")
                f.write(f"Changes:\n")

                # Handle architecture changes which may be a list of dictionaries
                if isinstance(change['recommendations']['changes'], list):
                    for item in change['recommendations']['changes']:
                        for k, v in item.items():
                            f.write(f"- {k}: {v}\n")
                else:
                    for k, v in change['recommendations']['changes'].items():
                        f.write(f"- {k}: {v}\n")

                f.write("\n")

        # Write hyperparameter changes
        if hp_changes:
            f.write("## Hyperparameter Changes\n\n")
            for change in hp_changes:
                f.write(f"**Epoch {change['epoch']}**\n\n")
                f.write(f"Explanation: {change['recommendations']['explanation']}\n\n")
                f.write(f"Changes:\n")

                # Handle hyperparameter changes which are usually a dictionary
                for k, v in change['recommendations']['changes'].items():
                    f.write(f"- {k}: {v}\n")

                f.write("\n")


# Define the state schema
class OptimizationState(TypedDict):
    model_config: Dict[str, Any]
    current_stats: Dict[str, Any]
    training_history: List[Dict[str, Any]]
    recommendations_history: List[Dict[str, Any]]
    model_changes_history: List[Dict[str, Any]]
    current_epoch: int
    total_epochs: int
    model_version: int
    action: str  # What to do next
    apply_changes: bool  # Whether to apply the suggested changes

# Define the nodes in our graph
def analyze_model_health(state: OptimizationState) -> OptimizationState:
    """Comprehensive analysis of model health, including gradients and stability"""
    print("""Analyzing gradient patterns and overall model stability...""")
    bedrock = boto3.client(service_name='bedrock-runtime', region_name=state['model_config']['aws_region'])

    prompt = f"""
    Perform a comprehensive analysis of this neural network's health and stability. The main objective is to improve
    the validation accuracy of the model.

    Model config:
    {json.dumps(state['model_config'], indent=2)}

    Current training stats:
    {json.dumps(state['current_stats'], indent=2)}

    Training history:
    {json.dumps(state['training_history'][-15:] if len(state['training_history']) > 5 else state['training_history'], indent=2)}

    Model changes history:
    {json.dumps(state['model_changes_history'][-5:] if len(state['model_changes_history']) > 5 else state['model_changes_history'], indent=2)}

    Part 1: Gradient Analysis
    ------------------------
    1. Are there signs of vanishing or exploding gradients?
    2. Which layers show abnormal gradient behavior?
    3. How have gradients evolved over the training process?
    4. Are there architectural bottlenecks causing gradient issues?

    Part 2: Overall Stability Assessment
    ----------------------------------
    1. Is the loss consistently decreasing?
    2. Is the validation accuracy improving?
    3. Have recent changes shown diminishing returns?
    4. Is the model stable enough to continue training without further changes?

    Return a JSON object with these sections:

    {{
        "gradient_analysis": {{
            "gradient_issues": [list of specific gradient problems identified],
            "affected_layers": [layers most impacted by these issues],
            "severity": integer from 1-5 with 5 being most severe,
            "insights": "deeper analysis of why these issues are occurring"
        }},
        "stability_assessment": {{
            "is_stable": boolean - whether the model is stable enough to complete training without further changes,
            "stability_score": integer from 1-10 where 10 is completely stable,
            "explanation": "detailed explanation of stability assessment",
            "remaining_issues": [list of any issues that still exist but don't necessarily require changes],
            "recommendation": "architecture" or "hyperparameters" - if not stable, what type of change is most needed
        }}
    }}
    """


    response_body = call_bedrock_with_tenacity(
                                            bedrock_client=bedrock,
                                            model_id=CROSS_REGION_ARN,
                                            prompt=prompt,
                                            max_tokens=2048,
                                            #thinking={"type": "enabled", "budget_tokens": 2048}
                                        )

    #response_body = json.loads(response['body'].read())
    analysis = response_body['content'][0]['text']

    # Extract JSON from the response
    try:
        analysis_json = json.loads(analysis)
    except:
        json_match = re.search(r'```json(.+?)```', analysis, re.DOTALL)
        if json_match:
            analysis_json = json.loads(json_match.group(1).strip())
        else:
            # Fallback if JSON not found
            analysis_json = {
                "gradient_analysis": {
                    "gradient_issues": ["Unable to parse structured analysis"],
                    "affected_layers": [],
                    "severity": 3,
                    "insights": "Parse error occurred extracting analysis"
                },
                "stability_assessment": {
                    "is_stable": False,
                    "stability_score": 5,
                    "explanation": "Unable to parse stability analysis",
                    "remaining_issues": ["Unable to parse analysis"],
                    "recommendation": "architecture"
                }
            }

    # Update state with analysis
    new_state = state.copy()
    new_state['current_stats']['gradient_analysis'] = analysis_json['gradient_analysis']
    new_state['current_stats']['stability_analysis'] = analysis_json['stability_assessment']

    # Log stability analysis
    logger.info(f"Stability Analysis:")
    logger.info(f"Is Stable: {analysis_json['stability_assessment'].get('is_stable', False)}")
    logger.info(f"Stability Score: {analysis_json['stability_assessment'].get('stability_score', 0)}/10")
    logger.info(f"Explanation: {analysis_json['stability_assessment'].get('explanation', 'No explanation')}")

    # Also write to a specific file
    with open(f"./logs/stability_analysis_v{state['model_version']}_e{state['current_epoch']}.txt", "w") as f:
        f.write(f"STABILITY ANALYSIS\n")
        f.write(f"=================\n\n")
        f.write(f"Is Stable: {analysis_json['stability_assessment'].get('is_stable', False)}\n")
        f.write(f"Stability Score: {analysis_json['stability_assessment'].get('stability_score', 0)}/10\n")
        f.write(f"Explanation: {analysis_json['stability_assessment'].get('explanation', 'No explanation')}\n\n")
        f.write(f"Remaining Issues:\n")
        for issue in analysis_json['stability_assessment'].get('remaining_issues', []):
            f.write(f"- {issue}\n")

    # Set action for router
    new_state['action'] = "analyze_model_health"
    return new_state


def extract_model_config(text):
    text = text.replace('json','')

    # Create a dictionary to store the extracted values
    result = {}
    content = text

    # Extract explanation text
    explanation_match = re.search(r'"explanation":\s*"([\s\S]*?)",\s*"changes":', content)
    if explanation_match:
        result['explanation'] = explanation_match.group(1)

    # Extract changes as text, being careful with nested braces
    changes_match = re.search(r'"changes":\s*(\{[\s\S]*?\})', content)
    if changes_match:
        changes_text = changes_match.group(1)

        # Count braces to ensure proper JSON structure
        open_braces = changes_text.count('{')
        close_braces = changes_text.count('}')

        # Adjust if there are too many closing braces
        if close_braces > open_braces:
            changes_text = changes_text[:-(close_braces-open_braces)]

        try:
            # Try to parse changes as JSON
            result['changes'] = json.loads(changes_text)
        except json.JSONDecodeError:
            # Manual parsing as a fallback
            changes_dict = {}

            # Extract key-value pairs with regex
            key_value_pairs = re.findall(r'"([^"]+)":\s*([\d\.]+|"[^"]*"|true|false)', changes_text)
            for key, value in key_value_pairs:
                # Convert value to appropriate type
                if value.lower() == 'true':
                    changes_dict[key] = True
                elif value.lower() == 'false':
                    changes_dict[key] = False
                elif value.replace('.', '', 1).isdigit():
                    changes_dict[key] = float(value) if '.' in value else int(value)
                else:
                    # Remove quotes if it's a string
                    changes_dict[key] = value.strip('"')

            result['changes'] = changes_dict

    return result


def modify_architecture(state: OptimizationState) -> OptimizationState:


    """Recommends and generates a new CNN architecture file with validation"""
    print("""Analyzing gradients and generating new CNN architecture file...""")

    # Ensure the model versions directory exists
    model_versions_dir = './model_versions'
    os.makedirs(model_versions_dir, exist_ok=True)

    # Create a client for bedrock
    bedrock = boto3.client(service_name='bedrock-runtime',
                           region_name=state['model_config']['aws_region'])

    # Get previous architecture changes if any
    previous_arch_changes = [
        change for change in state['model_changes_history']
        if change['type'] == 'architecture'
    ]

    # Current model version and new version
    current_version = state['model_config'].get('model_version', 1)
    new_version = current_version + 1

    # Get the current model file path
    if current_version == 1:
        current_model_file = 'cnn_model.py'  # Base model
    else:
        current_model_file = f'./model_versions/cnn_model_v{current_version}.py'

    # Read the current model file content
    try:
        with open(current_model_file, 'r') as f:
            current_model_code = f.read()
    except FileNotFoundError:
        # Fall back to base model if file not found
        with open('cnn_model.py', 'r') as f:
            current_model_code = f.read()

    # Create prompt for architecture improvement
    prompt = f"""
    Your task is to analyze the current CNN architecture and create an improved version based on gradient analysis.
    The model objective is the classification of images that are 32x32x3.

    Here's the gradient analysis data:
    {json.dumps(state['current_stats'].get('gradient_analysis', {}), indent=2)}

    Training history:
    {json.dumps(state['training_history'][-3:] if len(state['training_history']) > 3 else state['training_history'], indent=2)}

    Current model configuration:
    {json.dumps(state['model_config'], indent=2)}

    Previous architecture changes:
    {json.dumps(previous_arch_changes[-2:] if len(previous_arch_changes) > 2 else previous_arch_changes, indent=2)}

    Here's the current CNN model code:
    ```python
    {current_model_code}
    ```

    Please create an improved version of the CNN architecture to address the gradient issues identified.

    Your output should be a complete Python file with the improved CNN class that:
    1. Maintains the same class name (CNN) and basic interface
    2. Specifically addresses gradient flow issues
    3. Can be directly imported and used as a drop-in replacement
    4. Includes detailed comments explaining your changes and why they should help
    5. Adds a VERSION constant at the top set to {new_version}
    6. If you use a custom loss function, name it "loss_function" as member function of the object.

    Return ONLY the Python code for the new model file, without any additional text, markdown, or code formatting.
    """


    print("Reasoning, give me time to think...")
    model_code = call_bedrock_with_tenacity(
                                            bedrock_client=bedrock,
                                            model_id=CROSS_REGION_ARN,
                                            prompt=prompt,
                                            max_tokens=20000,
                                            thinking={"type": "enabled", "budget_tokens": 2048}
                                        )

    # Clean up the code (remove any markdown code block formatting if present)
    if "```python" in model_code:
        model_code = re.search(r'```python\s*(.*?)\s*```', model_code, re.DOTALL).group(1)

    # New reflection step-------------------------------------------------------------
    reflection_prompt = f"""
    Examine the CNN model code you've just generated:

    ```python
    {model_code}
    ```

    Please validate this architecture carefully, focusing on:

    1. SKIP CONNECTION VALIDATION:
       - Ensure tensors have matching dimensions (H,W,C) before addition
       - Check channel counts match at connection points
       - Validate spatial dimensions after pooling operations

    2. TENSOR SHAPE TRACKING:
       - Track tensor shapes through each layer
       - Consider effects of stride, padding, and pooling
       - Verify tensor shapes remain consistent throughout

    3. IMPLEMENTATION FIXES:
       - If you detect any shape mismatches or potential runtime errors, fix them
       - For skip connections with channel mismatches, add 1x1 convolutions to match channels
       - For spatial dimension mismatches, add appropriate padding or use adaptive operations

    Return the FIXED Python code that addresses all issues. Include comments explaining your changes.
    """

    validated_code = call_bedrock_with_tenacity(
                                            bedrock_client=bedrock,
                                            model_id=CROSS_REGION_ARN,
                                            prompt=reflection_prompt,
                                            max_tokens=20000,
                                            thinking={"type": "enabled", "budget_tokens": 2048}
                                        )

    # Clean up and extract code
    if "```python" in validated_code:
        validated_code = re.search(r'```python\s*(.*?)\s*```', validated_code, re.DOTALL).group(1)

    # Add an execution test to verify the model runs
    def test_model(code):
        temp_filename = f"temp_model_test_{uuid.uuid4()}.py"
        try:
            with open(temp_filename, 'w') as f:
                f.write(code)

            # Import the model
            spec = importlib.util.spec_from_file_location("temp_model", temp_filename)
            temp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_module)

            # Create model and test with sample input
            test_model = temp_module.CNN()
            test_input = torch.randn(2, 3, 32, 32)  # Sample input for CIFAR
            with torch.no_grad():
                output = test_model(test_input)

            return True, None
        except Exception as e:
            return False, str(e)
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    # Test execution
    success, error = test_model(validated_code)

    # If test fails, try to fix
    if not success:
        fix_prompt = f"""
        The CNN architecture you generated has runtime errors when executed:

        ERROR: {error}

        Original code:
        ```python
        {validated_code}
        ```

        Please fix these errors. Common issues include:
        - Skip connection dimension mismatches
        - Channel count mismatches
        - Incorrect size calculations after pooling

        Provide a fixed version that will execute without errors on a 32x32x3 input tensor.
        """

        fix_response = bedrock.invoke_model(
            modelId= CROSS_REGION_ARN,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4000,
                "messages": [{"role": "user", "content": fix_prompt}]
            })
        )

        fix_body = json.loads(fix_response['body'].read())
        fixed_code = fix_body['content'][0]['text']

        if "```python" in fixed_code:
            fixed_code = re.search(r'```python\s*(.*?)\s*```', fixed_code, re.DOTALL).group(1)

        # Test the fixed code
        success, _ = test_model(fixed_code)
        if success:
            validated_code = fixed_code
            logger.info("Architecture fixed after error correction")

    # Save the validated code
    model_code = validated_code

    # [rest of existing function to save the model file]

    # Ensure the code has the version number
    if "VERSION = " not in model_code:
        model_code = f"VERSION = {new_version}\n\n{model_code}"

    # Save the new model to file
    new_model_file = f'./model_versions/cnn_model_v{new_version}.py'
    with open(new_model_file, 'w') as f:
        f.write(model_code)

    logger.info(f"Generated new CNN model version {new_version} saved to {new_model_file}")

    # Extract explanation from the code
    explanation_match = re.search(r'"""(.*?)"""', model_code, re.DOTALL)
    explanation = explanation_match.group(1).strip() if explanation_match else "No explicit explanation found in model code."

    # Find changes by comparing parts of the code
    # This is a simple heuristic - in practice, you might want more sophisticated analysis
    changes = {
        "model_version": new_version,
        "model_file": new_model_file,
    }

    # Add specific changes detected
    if "use_skip_connections" in model_code and "use_skip_connections" not in current_model_code:
        changes["add_skip_connections"] = True

    if "BatchNorm" in model_code and "BatchNorm" not in current_model_code:
        changes["add_batch_norm"] = True

    if "dropout_rate" in model_code:
        # Try to extract the default dropout rate
        dropout_match = re.search(r'dropout_rate=([0-9\.]+)', model_code)
        if dropout_match:
            changes["dropout_rate"] = float(dropout_match.group(1))

    # Update state with recommendation
    new_state = state.copy()

    # Create recommendation entry
    rec_entry = {
        "type": "architecture",
        "epoch": state['current_epoch'],
        "recommendations": {
            "explanation": explanation,
            "changes": changes
        },
        "full_text": model_code
    }
    new_state['recommendations_history'].append(rec_entry)

    # Add to model changes history
    change_entry = {
        "type": "architecture",
        "epoch": state['current_epoch'],
        "changes": changes,
        "explanation": explanation
    }
    new_state['model_changes_history'].append(change_entry)

    # Update model config
    new_state['model_config']['model_version'] = new_version
    new_state['model_config']['model_file'] = new_model_file

    # Set action
    new_state['action'] = "modify_architecture"
    return new_state

def modify_hyperparameters(state: OptimizationState) -> OptimizationState:
    print("""Recommending and preparing hyperparameter adjustments based on gradient analysis""")
    bedrock = boto3.client(service_name='bedrock-runtime', region_name=state['model_config']['aws_region'])

    # Get previous hyperparameter changes if any
    previous_hp_changes = [
        change for change in state['model_changes_history']
        if change['type'] == 'hyperparameters'
    ]

    # Get previous recommendations
    previous_hp_recommendations = [
        rec for rec in state['recommendations_history']
        if rec['type'] == 'hyperparameters'
    ]

    prompt = f"""
    Based on the gradient analysis and training progress, you need to recommend and implement specific hyperparameter adjustments to improve training.

    Current Model Config:
    {json.dumps(state['model_config'], indent=2)}

    Gradient Analysis:
    {json.dumps(state['current_stats'].get('gradient_analysis', {}), indent=2)}

    Training History:
    {json.dumps(state['training_history'][-3:] if len(state['training_history']) > 3 else state['training_history'], indent=2)}

    Previous Hyperparameter Changes:
    {json.dumps(previous_hp_changes[-2:] if len(previous_hp_changes) > 2 else previous_hp_changes, indent=2)}

    Your task is to modify the hyperparameters to address the gradient issues. You need to provide exact, implementable changes.

    Return a JSON object with these fields:
    - "explanation": A detailed explanation of your hyperparameter changes and why they should help
    - "changes": A dictionary with these possible keys (include only those you want to change):
        - "learning_rate": new learning rate value (float)
        - "momentum": new momentum value (float)
        - "weight_decay": new weight decay value (float)
        - "batch_size": new batch size value (int)
        - "optimizer": optimizer type (string: "SGD", "Adam", "AdamW", "RMSprop")
        - "lr_schedule": learning rate schedule type (string: "step", "cosine")
        - "lr_steps": list of epochs for step schedule, e.g. [30, 60, 80]
        - "lr_gamma": learning rate reduction factor (float)
        - "gradient_clip": gradient clipping value (float)

    Please be precise in your changes. All values should be valid Python literals. Ensure your recommendations are technically sound and address the specific gradient issues identified.
    """

    response = bedrock.invoke_model(
        modelId=state['model_config']['claude_model_id'],
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1200,
            "messages": [{"role": "user", "content": prompt}]
        })
    )

    response_body = json.loads(response['body'].read())
    recommendations = response_body['content'][0]['text']

    try:
        recommendations_json = json.loads(recommendations)
    except:
        # Extract JSON
        json_match = re.search(r'```json(.+?)```', recommendations, re.DOTALL)
        if json_match:
            try:
                recommendations_json = json.loads(json_match.group(1).strip())
            except:
                recommendations_json = {
                    "explanation": "Error parsing JSON response",
                    "changes": {}
                }
        else:
            recommendations_json = extract_model_config(recommendations)


    # Update state with recommendations
    new_state = state.copy()
    rec_entry = {
        "type": "hyperparameters",
        "epoch": state['current_epoch'],
        "recommendations": recommendations_json,
        "full_text": recommendations
    }
    new_state['recommendations_history'].append(rec_entry)

    # Create hyperparameter changes to be applied
    hyperparameter_changes = {}
    if "changes" in recommendations_json:
        changes = recommendations_json["changes"]

        # Extract valid hyperparameter changes
        for key in ["learning_rate", "momentum", "weight_decay", "batch_size",
                   "optimizer", "lr_schedule", "lr_steps", "lr_gamma", "gradient_clip"]:
            if key in changes:
                hyperparameter_changes[key] = changes[key]

    # Add to model changes history
    if hyperparameter_changes:
        change_entry = {
            "type": "hyperparameters",
            "epoch": state['current_epoch'],
            "changes": hyperparameter_changes,
            "explanation": recommendations_json.get("explanation", "No explanation provided")
        }
        new_state['model_changes_history'].append(change_entry)

    new_state['action'] = "modify_hyperparameters"
    return new_state

def generate_summary(state: OptimizationState) -> OptimizationState:
    print("""Generate a comprehensive summary and final recommendations""")
    bedrock = boto3.client(service_name='bedrock-runtime', region_name=state['model_config']['aws_region'])

    prompt = f"""
    Create a comprehensive summary of the neural network optimization process.

    Current model config:
    {json.dumps(state['model_config'], indent=2)}

    Training history:
    {json.dumps(state['training_history'][-5:] if len(state['training_history']) > 5 else state['training_history'], indent=2)}

    Model changes history:
    {json.dumps(state['model_changes_history'][-5:] if len(state['model_changes_history']) > 5 else state['model_changes_history'], indent=2)}

    Please provide:
    1. A summary of the training progression
    2. Key gradient issues identified and how they evolved
    3. The architectural and hyperparameter changes made and their impact
    4. Analysis of which changes were most effective
    5. Final advice for future training iterations
    6. An assessment of overall neural network health
    """

    response = bedrock.invoke_model(
        modelId=state['model_config']['claude_model_id'],
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}]
        })
    )

    response_body = json.loads(response['body'].read())
    summary = response_body['content'][0]['text']

    # Update state
    new_state = state.copy()
    new_state['current_stats']['final_summary'] = summary

    # Save summary to file
    summary_filename = f"./logs/training_summary_v{state['model_version']}_epoch_{state['current_epoch']}.txt"
    with open(summary_filename, "w") as f:
        f.write(summary)

    print("\nFINAL TRAINING SUMMARY:")
    print("-" * 80)
    print(summary)
    print("-" * 80)

    new_state['action'] = "generate_summary"
    return new_state


def fix_autograd_error(model, error_msg, traceback_info, device, args):
    """
    Generate a fixed model version that resolves autograd errors related to inplace operations

    Args:
        model: The model that encountered the error
        error_msg: The error message string
        traceback_info: The traceback string
        device: The device the model is running on
        args: Training arguments dictionary

    Returns:
        tuple: (fixed_model, new_optimizer, success)
    """
    logger.info("Attempting to fix autograd errors in model architecture...")

    # Ensure the model versions directory exists
    model_versions_dir = './model_versions'

    # Get current model version by scanning the model_versions directory
    current_version = 1  # Start with base model as default
    if os.path.exists(model_versions_dir):
        #version_pattern = re.compile(r'cnn_model_v(\d+)\.py\$')
        version_pattern = re.compile(r'cnn_model_v(\d+)\.py')
        for filename in os.listdir(model_versions_dir):
            match = version_pattern.match(filename)
            if match:
                try:
                    version = int(match.group(1))
                    if version > current_version:
                        current_version = version
                except ValueError:
                    pass

    # New version will be current + 1
    new_version = current_version + 1

    # Get the current model file
    if current_version == 1:
        current_model_file = 'cnn_model.py'  # Base model
    else:
        current_model_file = f'./model_versions/cnn_model_v{current_version}.py'

    # Read the current model file content
    try:
        with open(current_model_file, 'r') as f:
            current_model_code = f.read()
    except FileNotFoundError:
        # Fall back to base model if file not found
        with open('cnn_model.py', 'r') as f:
            current_model_code = f.read()

    # Extract model architecture string representation
    model_arch = str(model)

    # Create a client for bedrock
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1'
    )



    # Create prompt to fix the model
    prompt = f"""
    I need you to fix a PyTorch CNN model that is encountering an autograd error during backpropagation.

    ERROR MESSAGE:
    ```
    {error_msg}
    ```

    ERROR TRACEBACK:
    ```
    {traceback_info}
    ```

    MODEL ARCHITECTURE:
    ```
    {model_arch}
    ```

    CURRENT MODEL CODE:
    ```python
    {current_model_code}
    ```

    This is most likely related to in-place operations in the model that are causing issues with the backward pass.
    Common problems include:

    1. Using in-place operations like x += y or operations with inplace=True
    2. ReLU layers with inplace=True which modify tensors needed for gradient computation
    3. Modifying tensors in place in custom modules that are needed for gradients
    4. Issues with skip connections or residual blocks

    Please create a fixed version of this CNN model that:
    1. Eliminates all in-place operations that could interfere with autograd
    2. Maintains the same overall architecture and functionality
    3. Ensures all operations preserve the computation graph for backpropagation
    4. Adds VERSION = {new_version} at the top of the file

    Return ONLY the complete Python code for the fixed model without additional text or code blocks.
    """

    try:
        # Get fixed model code from Bedrock
        model_code = call_bedrock_with_tenacity(
            bedrock_client=bedrock,
            model_id=CROSS_REGION_ARN,
            prompt=prompt,
            max_tokens=20000,
            #thinking={"type": "enabled", "budget_tokens": 2048}
        )

        save = model_code.copy()
        model_code = model_code['content'][0]['text']

        # Clean up the code (remove any markdown code block formatting if present)
        if "```python" in model_code:
            model_code = re.search(r'```python\s*(.*?)\s*```', model_code, re.DOTALL).group(1)

        # Ensure the code has the version number
        if "VERSION = " not in model_code:
            model_code = f"VERSION = {new_version}\n\n{model_code}"

        # Test the model to ensure it works
        def test_model(code):
            temp_filename = f"temp_model_test_{uuid.uuid4()}.py"
            try:
                with open(temp_filename, 'w') as f:
                    f.write(code)

                # Import the model
                spec = importlib.util.spec_from_file_location("temp_model", temp_filename)
                temp_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(temp_module)

                # Create model and test with sample input
                test_model = temp_module.CNN().to(device)
                test_input = torch.randn(2, 3, 32, 32).to(device)
                test_target = torch.randint(0, 10, (2,)).to(device)

                # Test forward pass
                output = test_model(test_input)

                # Test backward pass
                if hasattr(test_model, 'loss_function'):
                    loss = test_model.loss_function(output, test_target)
                else:
                    loss = nn.CrossEntropyLoss()(output, test_target)
                loss.backward()

                return True, None
            except Exception as e:
                return False, str(e)
            finally:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

        # Test execution
        success, error = test_model(model_code)

        if not success:
            # If test fails, try to fix again
            fix_prompt = f"""
            The fixed model still has errors when tested:

            ERROR: {error}

            CODE:
            ```python
            {model_code}
            ```

            Please fix these errors and ensure there are NO in-place operations that could interfere with autograd.
            """

            fix_model_code = call_bedrock_with_tenacity(
                bedrock_client=bedrock,
                model_id=CROSS_REGION_ARN,
                prompt=fix_prompt,
                max_tokens=10000
            )

            if "```python" in fix_model_code:
                fix_model_code = re.search(r'```python\s*(.*?)\s*```', fix_model_code, re.DOTALL).group(1)

            # Test the fixed code
            success, _ = test_model(fix_model_code)
            if success:
                model_code = fix_model_code
                logger.info("Model fixed after error correction")

        # Save the new model to file if successful
        if success:
            new_model_file = f'./model_versions/cnn_model_v{new_version}.py'
            with open(new_model_file, 'w') as f:
                f.write(model_code)

            logger.info(f"Generated fixed CNN model version {new_version} saved to {new_model_file}")

            # Update the model config
            args['model_config']['model_version'] = new_version
            args['model_config']['model_file'] = new_model_file

            # Load the new model
            spec = importlib.util.spec_from_file_location(
                f"cnn_model_v{new_version}",
                new_model_file
            )
            cnn_module = importlib.util.module_from_spec(spec)
            sys.modules[f"cnn_model_v{new_version}"] = cnn_module
            spec.loader.exec_module(cnn_module)

            # Create new model instance
            fixed_model = cnn_module.CNN().to(device)

            # Create new optimizer for fixed model
            optimizer_type = args['model_config'].get('optimizer', 'SGD')
            lr = args['model_config'].get('learning_rate', 0.01)
            momentum = args['model_config'].get('momentum', 0.9)
            weight_decay = args['model_config'].get('weight_decay', 1e-4)

            if optimizer_type == 'SGD':
                new_optimizer = optim.SGD(fixed_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            elif optimizer_type == 'Adam':
                new_optimizer = optim.Adam(fixed_model.parameters(), lr=lr, weight_decay=weight_decay)
            elif optimizer_type == 'AdamW':
                new_optimizer = optim.AdamW(fixed_model.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                new_optimizer = optim.SGD(fixed_model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

            return fixed_model, new_optimizer, True
        else:
            return None, None, False

    except Exception as fix_error:
        logger.error(f"Failed to fix autograd error: {fix_error}")
        return None, None, False


# Define the router function to determine next step
def router(state: OptimizationState) -> str:
    """Determine the next node in the graph based on state"""
    if state["action"] == "analyze_model_health":
        # Get stability analysis
        stability_analysis = state['current_stats'].get('stability_analysis', {})
        is_stable = stability_analysis.get('is_stable', False)

        # If model is at the end of training, generate summary
        if state["current_epoch"] == state["total_epochs"] - 1:
            return "generate_summary"

        # If model is stable, end the LLM optimization cycle
        if is_stable:
            logger.info("Model is considered STABLE. Will continue training without changes.")
            return END

        # If not stable, get recommendation from stability analysis
        recommend = stability_analysis.get('recommendation', 'architecture')

        # Route based on recommendation
        if recommend.lower() == 'hyperparameters':
            logger.info("Stability analysis recommends hyperparameter changes")
            return "modify_hyperparameters"
        else:
            # Default to architecture changes if not specified or if architecture is recommended
            logger.info("Stability analysis recommends architecture changes")
            return "modify_architecture"

    elif state["action"] == "modify_architecture":
        # After modifying architecture, just end the cycle
        return END

    elif state["action"] == "modify_hyperparameters":
        # After modifying hyperparameters, just end the cycle
        return END

    elif state["action"] == "generate_summary":
        return END

    else:
        # Default case to avoid KeyError
        logger.warning(f"Unknown action '{state['action']}', defaulting to END")
        return END


# Set up argument parser for hyperparameters
def parse_args():
    parser = argparse.ArgumentParser(description='CNN with gradient norm tracking and LLM optimization')

    # Model architecture
    parser.add_argument('--num_conv_layers', type=int, default=10, help='Number of convolutional layers')
    parser.add_argument('--filters', type=str, default='64,128,256,512,512,512,512,512,512,512', help='Number of filters per layer, comma-separated')
    parser.add_argument('--kernel_size', type=int, default=15, help='Kernel size for convolutions')
    parser.add_argument('--fc_dims', type=str, default='2,1', help='Dimensions of fully connected layers, comma-separated')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate for FC layers')
    parser.add_argument('--use_skip_connections', action='store_true', help='Use skip connections in architecture')
    parser.add_argument('--use_batch_norm', action='store_true', help='Use batch normalization')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=512, help='Input batch size')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 penalty)')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--lr_schedule', type=str, default='step', choices=['step', 'cosine'], help='Learning rate schedule')
    parser.add_argument('--lr_steps', type=str, default='30,60,80', help='Epochs at which to reduce LR (for step schedule)')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='LR reduction factor')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW', 'RMSprop'], help='Optimizer')
    parser.add_argument('--gradient_clip', type=float, default=None, help='Gradient clipping value')

    # Data and infrastructure
    parser.add_argument('--data_path', type=str, default='./data/imagenet', help='Path to ImageNet data')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--log_interval', type=int, default=10, help='How many batches to wait before logging training status')

    # AWS Bedrock settings
    parser.add_argument('--aws_region', type=str, default='us-east-1', help='AWS region')
    parser.add_argument('--claude_model_id', type=str, default='anthropic.claude-3-sonnet-20240229-v1:0', help='Claude model to use')
    parser.add_argument('--analyze_interval', type=int, default=50, help='How many epochs to wait before asking Claude for analysis')

    # Model evolution settings
    parser.add_argument('--apply_changes', action='store_true', default=True, help='Apply LLM recommended changes to model')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--transfer_weights', action='store_true', help='Transfer weights when architecture changes')


    args = parser.parse_args()

    # Process string arguments into lists
    args.filters = [int(f) for f in args.filters.split(',')]
    args.fc_dims = [int(d) for d in args.fc_dims.split(',')]
    args.lr_steps = [int(s) for s in args.lr_steps.split(',')]

    return args


# Function to calculate gradient norm for each layer and total
def get_gradient_norms(model):
    total_norm = 0
    layer_norms = {}

    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            layer_norms[name] = param_norm.item()
            total_norm += param_norm.item() ** 2

    total_norm = total_norm ** 0.5
    return total_norm, layer_norms

# Setup data transforms and loaders

def setup_data(args, rank, world_size, quick_test=True, num_samples=100):
    """
    Set up data loaders with option for quick testing with reduced dataset

    Args:
        args: Command line arguments
        rank: Process rank for distributed training
        quick_test: If True, use only a small subset of data
        num_samples: Number of samples to use when quick_test is True
    """
    # Data transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Load dataset (using CIFAR-10)
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=val_transform)

    # If quick test is enabled, select only a subset of images
    if quick_test:

        # Create random indices but make sure we get some from each class for balance
        train_indices = []
        val_indices = []

        # CIFAR-10 has 10 classes, try to get even representation
        samples_per_class = max(num_samples // 10, 1)  # at least 1 sample per class

        # Get indices for each class in training set
        train_targets = np.array(train_dataset.targets)
        for class_idx in range(10):
            class_indices = np.where(train_targets == class_idx)[0]
            # Randomly select samples_per_class indices for this class
            if len(class_indices) > samples_per_class:
                selected_indices = np.random.choice(class_indices, samples_per_class, replace=False)
                train_indices.extend(selected_indices)
            else:
                train_indices.extend(class_indices)

        # Get indices for each class in validation set
        val_targets = np.array(val_dataset.targets)
        val_samples_per_class = max(num_samples // 20, 1)  # Fewer validation samples
        for class_idx in range(10):
            class_indices = np.where(val_targets == class_idx)[0]
            # Randomly select val_samples_per_class indices for this class
            if len(class_indices) > val_samples_per_class:
                selected_indices = np.random.choice(class_indices, val_samples_per_class, replace=False)
                val_indices.extend(selected_indices)
            else:
                val_indices.extend(class_indices)

        # Create subset datasets
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

        print(f"Quick test enabled: Using {len(train_dataset)} training samples and {len(val_dataset)} validation samples")

    # Set up distributed samplers if needed
    if args.local_rank != -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                                                train_dataset,
                                                num_replicas=world_size,
                                                rank=rank,
                                                drop_last=False
                                                )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
                                            val_dataset,
                                            num_replicas=world_size,
                                            rank=rank,
                                            shuffle=False,
                                            drop_last=False
                                            )
    else:
        train_sampler = None
        val_sampler = None

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=min(args.batch_size, len(train_dataset)),
        shuffle=(train_sampler is None),
        #num_workers=args.workers,
        pin_memory=True, sampler=train_sampler
    )

    val_loader = DataLoader(
        val_dataset, batch_size=min(args.batch_size, len(val_dataset)),
        shuffle=False,
        #num_workers=args.workers,
        pin_memory=True, sampler=val_sampler
    )

    return train_loader, val_loader, train_sampler


# Setup model, optimizer, criterion and scheduler
def setup_model(model_config, device, previous_model=None, transfer_weights=False):
    """Setup model by loading the appropriate CNN version from file"""
    logger.info(f"Setting up model with config: {model_config}")

    # Determine which model version to load
    model_version = model_config.get('model_version', 1)
    model = None

    try:
        if model_version == 1:
            # Base model in main directory
            model_file = 'cnn_model'
            logger.info(f"Importing base CNN from {model_file}")
            cnn_module = importlib.import_module(model_file)
            importlib.reload(cnn_module)  # Ensure we get the latest version

        else:
            # Newer version in model_versions directory
            model_path = model_config.get('model_file', f'./model_versions/cnn_model_v{model_version}.py')
            model_path = os.path.abspath(model_path)
            logger.info(f"Attempting to load model from: {model_path}")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model_file = os.path.basename(model_path).replace('.py', '')
            spec = importlib.util.spec_from_file_location(model_file, model_path)
            cnn_module = importlib.util.module_from_spec(spec)
            sys.modules[model_file] = cnn_module
            spec.loader.exec_module(cnn_module)

        # Create model instance
        model = cnn_module.CNN()
        logger.info(f"Successfully loaded CNN model version {model_version}")

    except (ImportError, FileNotFoundError, AttributeError) as e:
        logger.error(f"Failed to load specified model version {model_version}: {e}")
        logger.info("Searching for available model versions...")

        # Look for available model versions
        available_models = []

        # Check for base model
        try:
            base_module = importlib.import_module('cnn_model')
            available_models.append((1, base_module, 'Base model'))
        except ImportError:
            pass

        # Check model_versions directory
        model_dir = './model_versions'
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.startswith('cnn_model_v') and file.endswith('.py'):
                    try:
                        ver_num = int(file.split('_v')[1].split('.py')[0])
                        file_path = os.path.join(model_dir, file)
                        mod_name = file.replace('.py', '')

                        spec = importlib.util.spec_from_file_location(mod_name, file_path)
                        mod = importlib.util.module_from_spec(spec)
                        sys.modules[mod_name] = mod
                        spec.loader.exec_module(mod)

                        available_models.append((ver_num, mod, file_path))
                    except Exception as e:
                        logger.warning(f"Could not load {file}: {e}")

        if available_models:
            # Sort by version number (highest first)
            available_models.sort(reverse=True)
            logger.info(f"Found {len(available_models)} available model versions.")

            # Use the highest available version
            version, cnn_module, path = available_models[0]
            logger.info(f"Using model version {version} from {path}")
            model = cnn_module.CNN()
        else:
            logger.error("No usable CNN model found. Cannot continue.")
            raise RuntimeError("No CNN model implementation available")

    if model_config['local_rank'] != -1:
       model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Move model to device
    model = model.to(device)

    # Wrap model with DDP if using distributed training
    local_rank = model_config.get('local_rank')
    if local_rank != -1:
        model = DDP(model, device_ids=[local_rank])

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    # Set up optimizer
    optimizer_type = model_config.get('optimizer', 'SGD')
    lr = model_config.get('learning_rate', 0.01)
    momentum = model_config.get('momentum', 0.9)
    weight_decay = model_config.get('weight_decay', 1e-4)

    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        logger.warning(f"Unknown optimizer {optimizer_type}, using SGD")
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Learning rate scheduler
    lr_schedule = model_config.get('lr_schedule', 'step')
    lr_steps = model_config.get('lr_steps', [30, 60, 80])
    lr_gamma = model_config.get('lr_gamma', 0.1)
    total_epochs = model_config.get('total_epochs', 90)

    if lr_schedule == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=lr_gamma)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    return model, criterion, optimizer, scheduler


# Apply architecture changes to model configuration
def apply_architecture_changes(model_config, architecture_changes):
    """Apply architecture changes to model configuration and return updated config"""
    new_config = copy.deepcopy(model_config)

    # Update configuration with changes
    for key, value in architecture_changes.items():
        if key in new_config:
            new_config[key] = value
        else:
            logger.info(f"Adding new configuration parameter: {key} = {value}")
            new_config[key] = value

    return new_config

# Apply hyperparameter changes to model configuration
def apply_hyperparameter_changes(model_config, hyperparameter_changes):
    """Apply hyperparameter changes to model configuration and return updated config"""
    new_config = copy.deepcopy(model_config)

    # Update configuration with changes
    for key, value in hyperparameter_changes.items():
        if key in new_config:
            new_config[key] = value
        else:
            logger.info(f"Adding new configuration parameter: {key} = {value}")
            new_config[key] = value

    return new_config

# Setup LangGraph workflow
def setup_langgraph(args, rank):
    if rank != 0:
        return None, None

    # Create the graph
    workflow = StateGraph(OptimizationState)

    # Add nodes - replace separate analysis nodes with the new combined one
    workflow.add_node("analyze_model_health", analyze_model_health)
    workflow.add_node("modify_architecture", modify_architecture)
    workflow.add_node("modify_hyperparameters", modify_hyperparameters)
    workflow.add_node("generate_summary", generate_summary)

    # Update the router to handle the new node
    workflow.add_conditional_edges(
        "analyze_model_health",
        router,
        {
            "modify_architecture": "modify_architecture",
            "modify_hyperparameters": "modify_hyperparameters",
            "generate_summary": "generate_summary",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "modify_architecture",
        router,
        {
            END: END
        }
    )

    workflow.add_conditional_edges(
        "modify_hyperparameters",
        router,
        {
            END: END
        }
    )

    workflow.add_edge("generate_summary", END)

    # Set the entry point to our new analysis function
    workflow.set_entry_point("analyze_model_health")

    # Setup checkpointer for persistence
    memory_saver = MemorySaver()

    # Compile the graph
    app = workflow.compile(checkpointer=memory_saver)
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return app, run_id



# Train for one epoch
def train_epoch(epoch, model, train_loader, criterion, optimizer, device, model_config):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_grad_norm_epoch = 0
    layer_grad_norms_epoch = defaultdict(float)

    # Track if we've already attempted to fix an autograd error
    autograd_fix_attempted = False

    # Apply gradient clipping if configured
    if model_config.get('gradient_clip') is not None:
        clip_value = model_config.get('gradient_clip')
        logger.info(f"Using gradient clipping with value {clip_value}")

    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{model_config['total_epochs']}") as t:
        batch_idx = 0
        while batch_idx < len(train_loader):
            try:
                # Get the next batch
                images, target = next(iter(train_loader))
                images, target = images.to(device), target.to(device)
                batch_idx += 1

                # Forward pass
                optimizer.zero_grad()
                output = model(images)
                if hasattr(model, 'loss_function'):
                    loss = model.loss_function(output, target)
                else:
                    loss = criterion(output, target)

                # Backward pass
                loss.backward()

                # Apply gradient clipping if configured
                if model_config.get('gradient_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), model_config.get('gradient_clip'))

                # Get gradient norms
                #if batch_idx % model_config['log_interval'] == 0:
                if model_config['local_rank'] != -1:
                    # For DDP, get gradients from the model's module
                    total_grad_norm, layer_grad_norms = get_gradient_norms(model.module)
                else:
                    total_grad_norm, layer_grad_norms = get_gradient_norms(model)

                total_grad_norm_epoch += total_grad_norm
                for layer, norm in layer_grad_norms.items():
                    layer_grad_norms_epoch[layer] += norm

                # Update weights
                optimizer.step()

                # Track statistics
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                # Update progress bar
                t.update(1)  # Explicitly update by 1
                t.set_postfix({
                    'loss': running_loss / batch_idx,
                    'acc': 100. * correct / total,
                })

            except RuntimeError as e:
                error_msg = str(e)
                if "inplace operation" in error_msg and not autograd_fix_attempted:
                    # This is an autograd error related to inplace operations
                    logger.error(f"Autograd error detected: {e}")

                    # Get the traceback
                    import traceback
                    tb_str = traceback.format_exc()

                    # Attempt to fix the issue by generating a new model
                    autograd_fix_attempted = True

                    # Try to fix the model
                    fixed_model, new_optimizer, success = fix_autograd_error(
                        model, error_msg, tb_str, device, model_config
                    )

                    if success:
                        # Replace the current model and optimizer with the fixed ones
                        model = fixed_model
                        optimizer = new_optimizer
                        logger.info("Successfully switched to fixed model!")

                        # Retry the current batch
                        batch_idx -= 1
                        continue
                    else:
                        logger.error("Failed to create a working fixed model")
                        raise e  # Re-raise the original error

                else:
                    # Re-raise the error if it's not an inplace operation issue or we've already tried to fix it
                    logger.error(f"Error during training: {e}")
                    raise

    # Normalize gradient norms by number of logged iterations
    num_logged = max(1, len(train_loader) // model_config['log_interval'])
    avg_total_grad_norm = total_grad_norm_epoch / num_logged
    avg_layer_grad_norms = {layer: norm / num_logged for layer, norm in layer_grad_norms_epoch.items()}

    epoch_loss = running_loss / batch_idx  # Use actual number of batches processed
    epoch_acc = 100. * correct / max(1, total)  # Avoid division by zero

    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'total_grad_norm': avg_total_grad_norm,
        'layer_grad_norms': avg_layer_grad_norms
    }






# Validate model
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, target in val_loader:
            images, target = images.to(device), target.to(device)
            output = model(images)
            loss = criterion(output, target)

            val_loss += loss.item()
            _, predicted = output.max(1)
            val_total += target.size(0)
            val_correct += predicted.eq(target).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100. * val_correct / val_total

    return {
        'loss': val_loss,
        'accuracy': val_acc
    }

# Run LangGraph analysis and apply changes
def run_analysis_and_apply_changes(app, run_id, model_config, current_stats, training_history,
                                  recommendations_history, model_changes_history,
                                  epoch, total_epochs, apply_changes):
    # Prepare input state for LangGraph workflow
    input_state = {
        "model_config": model_config,
        "current_stats": current_stats,
        "training_history": training_history,
        "recommendations_history": recommendations_history,
        "model_changes_history": model_changes_history,
        "current_epoch": epoch + 1,
        "total_epochs": total_epochs,
        "model_version": len(model_changes_history) + 1,
        "action": "analyze_gradients",
        "apply_changes": apply_changes
    }

    # Run the LangGraph workflow
    logger.info("\nRunning optimization workflow with LangGraph...")
    try:
        result = app.invoke(
            input_state,
            config={"configurable": {"thread_id": run_id}}
        )
        return result
    except Exception as e:
        logger.error(f"Error in LangGraph workflow: {str(e)}")
        # Return a fallback result to avoid crashing the training loop
        return {
            "recommendations_history": recommendations_history,
            "current_stats": current_stats,
            "model_changes_history": model_changes_history
        }

# Save model checkpoint
def save_checkpoint(model, optimizer, scheduler, model_config, epoch, stats, checkpoint_dir, is_final=False):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if is_final:
        filename = os.path.join(checkpoint_dir, f"final_model_v{model_config['model_version']}.pth")
    else:
        filename = os.path.join(checkpoint_dir, f"model_v{model_config['model_version']}_e{epoch}.pth")

    # Get model state dict (handle DDP case)
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'model_config': model_config,
        'stats': stats,
        'date': datetime.now().isoformat(),
    }

    torch.save(checkpoint, filename)
    logger.info(f"Saved checkpoint to {filename}")

    return filename

# Plot training statistics
def plot_stats(stats, model_version):
    plt.figure(figsize=(15, 15))

    plt.subplot(2, 2, 1)
    plt.plot(stats['epoch_losses'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')

    plt.subplot(2, 2, 2)
    plt.plot(stats['epoch_accuracies'])
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')

    plt.subplot(2, 2, 3)
    plt.plot(stats['total_grad_norms'])
    plt.title('Total Gradient Norm')
    plt.xlabel('Epoch')
    plt.ylabel('Norm')
    plt.yscale('log')

    plt.subplot(2, 2, 4)
    # Plot gradient norms for selected layers (first, middle, last conv and fc layers)
    layer_keys = list(stats['layer_grad_norms'].keys())
    conv_layers = [k for k in layer_keys if 'conv' in k]
    fc_layers = [k for k in layer_keys if 'fc' in k or 'classifier' in k]

    selected_layers = []
    if conv_layers:
        selected_layers.extend([conv_layers[0], conv_layers[len(conv_layers)//2], conv_layers[-1]])
    if fc_layers:
        selected_layers.extend([fc_layers[0], fc_layers[-1]])

    for layer in selected_layers:
        plt.plot(stats['layer_grad_norms'][layer], label=layer)

    plt.title('Layer Gradient Norms')
    plt.xlabel('Epoch')
    plt.ylabel('Norm')
    plt.yscale('log')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'training_stats_v{model_version}.png')
    #plt.show()
    plt.close()

#%%
# Main function
def main(rank, world_size):
##%%
    # Set up logging to file
    #logger = setup_logging()
    logger.info("Starting CNN gradient analyzer")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    # Parse arguments
    args = parse_args()

    # Detect available GPUs
    num_gpus = torch.cuda.device_count()
    logger.info(f"Found {num_gpus} GPUs")

    # Initialize the process group
    dist.init_process_group(backend='nccl',
                            rank=rank, world_size=world_size)

    # Initialize distributed training if required
    if args.local_rank != -1:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0

    # Create checkpoint directory
    if rank == 0 and not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # Set up data loaders
    train_loader, val_loader, train_sampler = setup_data(args, rank, world_size, quick_test=True, num_samples=10000)

    # Set up device
    device = torch.device(f"cuda:{rank}")

    # Convert args to model_config dictionary
    model_config = {
        'num_conv_layers': args.num_conv_layers,
        'filters': args.filters,
        'kernel_size': args.kernel_size,
        'fc_dims': args.fc_dims,
        'dropout_rate': args.dropout_rate,
        'learning_rate': args.learning_rate,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'optimizer': args.optimizer,
        'lr_schedule': args.lr_schedule,
        'lr_steps': args.lr_steps,
        'lr_gamma': args.lr_gamma,
        'use_skip_connections': args.use_skip_connections,
        'use_batch_norm': args.use_batch_norm,
        'gradient_clip': args.gradient_clip,
        'total_epochs': args.epochs,
        'local_rank': rank,
        'log_interval': args.log_interval,
        'claude_model_id': args.claude_model_id,
        'aws_region': args.aws_region,
        'model_version': 1  # Start with version 1
    }

    # Set up model, criterion, optimizer, scheduler
    model, criterion, optimizer, scheduler = setup_model(model_config, device)

    # Set up LangGraph workflow if rank is 0
    if rank == 0:
        app, run_id = setup_langgraph(args, rank)
    else:
        app, run_id = None, None

    # Training statistics
    stats = {
        'epoch_losses': [],
        'epoch_accuracies': [],
        'total_grad_norms': [],
        'layer_grad_norms': defaultdict(list),
        'val_losses': [],
        'val_accuracies': []
    }

    # Lists to store history for LangGraph
    training_history = []
    recommendations_history = []
    model_changes_history = []

    # Track model stability
    model_is_stable = False
    last_analysis_epoch = -1
    rebuild_model = False

##%%
    # Training loop
    for epoch in range(args.epochs):
        if args.local_rank != -1:
            train_sampler.set_epoch(epoch)

        # Train for one epoch
        train_stats = train_epoch(epoch, model, train_loader, criterion, optimizer, device, model_config)

        # Update learning rate
        scheduler.step()

        # Validate
        val_stats = validate(model, val_loader, criterion, device)

        # Update stats
        stats['epoch_losses'].append(train_stats['loss'])
        stats['epoch_accuracies'].append(train_stats['accuracy'])
        stats['total_grad_norms'].append(train_stats['total_grad_norm'])
        stats['val_losses'].append(val_stats['loss'])
        stats['val_accuracies'].append(val_stats['accuracy'])

        for layer, norm in train_stats['layer_grad_norms'].items():
            stats['layer_grad_norms'][layer].append(norm)

        # Print epoch statistics if rank is 0
        if rank == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}")
            logger.info(f"Train Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['accuracy']:.2f}%")
            logger.info(f"Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['accuracy']:.2f}%")
            logger.info(f"Total Grad Norm: {train_stats['total_grad_norm']:.4f}")
            logger.info(f"Current LR: {optimizer.param_groups[0]['lr']}")

            # Save checkpoint
            if epoch % 20 == 0 or epoch == args.epochs - 1 and epoch > 0:
                save_checkpoint(
                    model, optimizer, scheduler, model_config,
                    epoch, stats, args.checkpoint_dir,
                    is_final=(epoch == args.epochs - 1)
                )

            # Prepare epoch stats for LangGraph
            epoch_stats = {
                "epoch": epoch + 1,
                "train_loss": train_stats['loss'],
                "train_accuracy": train_stats['accuracy'],
                "val_loss": val_stats['loss'],
                "val_accuracy": val_stats['accuracy'],
                "total_grad_norm": train_stats['total_grad_norm'],
                "layer_grad_norms": {layer: norm for layer, norm in train_stats['layer_grad_norms'].items()}
            }

            # Add to training history
            training_history.append(epoch_stats)

            # Check if analysis should be run
            should_analyze = (
                # Run analysis based on interval if not yet stable
                #(not model_is_stable and (epoch - last_analysis_epoch) >= min(3, args.analyze_interval)) or
                (not model_is_stable and (epoch - last_analysis_epoch) >= args.analyze_interval) or
                # Run less frequently if model is stable
                (model_is_stable and (epoch - last_analysis_epoch) >= int(2*args.analyze_interval)) or
                # Always run at the end
                epoch == args.epochs - 1
            )

            # After specified intervals or at the end of training, run analysis
            rebuild_model = False
            if should_analyze:
                last_analysis_epoch = epoch
                logger.info(f"Running analysis at epoch {epoch+1} (model stability: {'stable' if model_is_stable else 'not yet stable'})")

                if app:  # Only if LangGraph is available
                    # Prepare current stats for LangGraph workflow
                    current_stats = {
                        "train_loss": train_stats['loss'],
                        "train_accuracy": train_stats['accuracy'],
                        "val_loss": val_stats['loss'],
                        "val_accuracy": val_stats['accuracy'],
                        "total_grad_norm": train_stats['total_grad_norm'],
                        "layer_grad_norms": {layer: train_stats['layer_grad_norms'][layer] for layer in train_stats['layer_grad_norms']}
                    }

                    # Run analysis and get recommendations
                    result = run_analysis_and_apply_changes(
                        app, run_id, model_config, current_stats,
                        training_history, recommendations_history, model_changes_history,
                        epoch, args.epochs, args.apply_changes
                    )

                    # Update histories from the result
                    if "recommendations_history" in result and result["recommendations_history"]:
                        recommendations_history = result["recommendations_history"]

                    # Check if model is now considered stable
                    if "current_stats" in result and "stability_analysis" in result["current_stats"]:
                        stability_analysis = result["current_stats"]["stability_analysis"]
                        model_is_stable = stability_analysis.get("is_stable", False)

                        if model_is_stable:
                            logger.info(f"Model is now considered STABLE (score: {stability_analysis.get('stability_score', 0)}/10)")
                            logger.info(f"Will continue training to completion without frequent changes")

                            # Write stability report
                            with open(f"./logs/model_stable_at_epoch_{epoch}.txt", "w") as f:
                                f.write(f"MODEL STABILITY ACHIEVED\n")
                                f.write(f"========================\n\n")
                                f.write(f"Epoch: {epoch + 1}/{args.epochs}\n")
                                f.write(f"Stability Score: {stability_analysis.get('stability_score', 0)}/10\n\n")
                                f.write(f"Explanation:\n{stability_analysis.get('explanation', 'No explanation provided')}\n\n")
                                f.write(f"Current Model Config:\n{json.dumps(model_config, indent=2)}\n")

                    if "model_changes_history" in result and result["model_changes_history"]:
                        # Update our history record
                        model_changes_history = result["model_changes_history"]

                        # Don't bother making changes if model is already stable
                        if not model_is_stable and args.apply_changes:
                            # Get the most recent changes of each type
                            latest_arch_change = next((change for change in recommendations_history
                                                    if change['type'] == 'architecture'), None)
                            latest_hp_change = next((change for change in recommendations_history
                                                    if change['type'] == 'hyperparameters'), None)

                            rebuild_model = False

                            # Apply architecture changes if we have any
                            if latest_arch_change:
                                specific_changes = latest_arch_change['recommendations']['changes']
                                if isinstance(specific_changes, list) and specific_changes:
                                    specific_changes=specific_changes[0]
                                #specific_changes = latest_arch_change['changes']
                                logger.info(f"Applying latest architecture changes: {specific_changes}")
                                model_config = apply_architecture_changes(model_config, specific_changes)
                                rebuild_model = True

                            # Apply hyperparameter changes if we have any
                            if latest_hp_change:
                                specific_changes = latest_hp_change['recommendations']['changes']
                                if isinstance(specific_changes, list) and specific_changes:
                                    specific_changes=specific_changes[0]
                                #specific_changes = latest_hp_change['changes']
                                logger.info(f"Applying latest hyperparameter changes: {specific_changes}")
                                model_config = apply_hyperparameter_changes(model_config, specific_changes)
                                rebuild_model = True

                            #model_changes_history = [latest_arch_change['recommendations'], latest_hp_change['recommendations']]
                            # Write changes summary to file after each analysis
                            #write_changes_summary(model_changes_history, f"./logs/model_changes_v{model_config['model_version']}.md")
                            write_changes_summary(recommendations_history, f"./logs/model_changes_v{model_config['model_version']}.md")

            if epoch % 2 == 0:
                plot_stats(stats, model_config['model_version'])

            # Rebuild model if needed and not stable
            if rebuild_model and args.apply_changes and not model_is_stable:
                logger.info("-" * 80)
                logger.info("REBUILDING MODEL WITH NEW CONFIGURATION")
                logger.info(f"Previous model config: {model_config}")
                print("\n\n\n\n\n\n\n\n\n\n\n")

                # Save old model for weight transfer
                old_model = model

                # Create new model
                model, criterion, optimizer, scheduler = setup_model(
                    model_config, device,
                    previous_model=old_model if args.transfer_weights else None,
                    transfer_weights=args.transfer_weights
                )

                # Save checkpoint of new model
                save_checkpoint(
                    model, optimizer, scheduler, model_config,
                    epoch, stats, args.checkpoint_dir,
                    is_final=False
                )

                logger.info(f"New model config: {model_config}")
                logger.info("-" * 80)
##%%
    # Save the final model (rank 0 only)
    if rank == 0:
        # Save final checkpoint
        save_checkpoint(model, optimizer, scheduler, model_config, args.epochs-1,
                      stats, args.checkpoint_dir, is_final=True)

        # Plot training statistics
        plot_stats(stats, model_config['model_version'])

        # Write final changes summary
        write_changes_summary(model_changes_history, f"final_model_changes_summary.md")

        # Save model evolution history
        evolution_history = {
            'training_history': training_history,
            'recommendations_history': recommendations_history,
            'model_changes_history': model_changes_history,
            'final_model_config': model_config,
            'model_is_stable': model_is_stable
        }

        with open(os.path.join(args.checkpoint_dir, './logs/model_evolution_history.json'), 'w') as f:
            json.dump(evolution_history, f, indent=2)

        logger.info("Training complete! Final model saved along with evolution history.")

#%% main starts
if __name__ == "__main__":

    ensure_empty_log_directory()
    ensure_empty_log_directory(log_dir="./model_versions")

    # I think because of langraph, lets keep this to 1 GPU for
    # the LLM and have ONLY_train.py handle the scaling to bigger data.
    main(0,1)
