"""
Generation and Evaluation Script for Multilingual Temporal Classification

This script provides two main functionalities:
1. Translate English questions to multiple languages using OpenAI API
2. Generate classifications using various LLMs via vLLM
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yaml
from tqdm import tqdm
from openai import OpenAI
from vllm import LLM, SamplingParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Translator:
    """Handles translation of questions using OpenAI API."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.target_languages = ["French", "Russian", "German", "Hebrew", "Arabic", "Chinese"]
    
    def create_translation_prompt(self, text: str) -> Dict:
        """Create a translation request."""
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": f"Translate the following English text into {', '.join(self.target_languages)}. "
                               f"Provide the translations as a JSON object with keys {self.target_languages}."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }
    
    def translate_batch(self, texts: List[str], batch_size: int = 10) -> List[Dict]:
        """Translate a batch of texts."""
        translations = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
            batch = texts[i:i + batch_size]
            batch_translations = []
            
            for text in batch:
                try:
                    response = self.client.chat.completions.create(
                        **self.create_translation_prompt(text)
                    )
                    result = json.loads(response.choices[0].message.content)
                    batch_translations.append(result)
                except Exception as e:
                    logger.error(f"Translation error for text: {text[:50]}... Error: {e}")
                    # Return empty translations on error
                    batch_translations.append({lang: "" for lang in self.target_languages})
            
            translations.extend(batch_translations)
        
        return translations


class Classifier:
    """Handles classification using vLLM."""
    
    # Few-shot examples for classification
    FEW_SHOT_EXAMPLES = """Examples

Question: What breed of dog is considered the smallest in the world? Classification: Mutable
Question: Which country is the largest producer of spices? Classification: Mutable
Question: In which city is the Australian Open tennis tournament held? Classification: Mutable
Question: What is the administrative division of Spain? Classification: Mutable
Question: What position does Ronaldo play? Classification: Mutable
Question: What was Cain's brother's name? Classification: Immutable
Question: Who is Galya Chertkova? Classification: Immutable
Question: How convenient is it to convert units of measurement from the Fahrenheit scale to degrees Celsius? Classification: Immutable
Question: What genes did modern humans inherit from Neanderthals? Classification: Immutable
Question: What is the tennis player's form? Classification: Immutable"""
    
    def __init__(self, model_name: str, tensor_parallel_size: int = 2):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.llm = None
        
    def initialize_model(self):
        """Initialize the vLLM model."""
        logger.info(f"Initializing model: {self.model_name}")
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=0.95,
            dtype='bfloat16',
            max_model_len=2000,
            trust_remote_code=True
        )
    
    def get_sampling_params(self) -> SamplingParams:
        """Get sampling parameters based on model type."""
        if "Qwen" in self.model_name:
            return SamplingParams(
                n=1, temperature=0.6, top_p=0.95, top_k=20, min_p=0,
                max_tokens=1536
            )
        else:
            return SamplingParams(
                n=1, temperature=0.6, top_p=0.9,
                max_tokens=512
            )
    
    def create_classification_prompt(self, question: str, use_deep_thinking: bool = False) -> str:
        """Create classification prompt."""
        deep_thinking = "/think " if use_deep_thinking else ""
        
        system_message = (
            f"{deep_thinking}You are a helpful assistant. You help user to classify "
            f"the questions based on the temporality. There are two classes: "
            f"immutable and mutable. Immutable, in which the answer almost never changes. "
            f"Mutable, in which the answer typically changes over the course of several "
            f"years or less. Think about each question and in the end answer with "
            f"Mutable or Immutable starting with 'Classification:'\n\n{self.FEW_SHOT_EXAMPLES}"
        )
        
        return f"{system_message}\n\nQuestion: {question}"
    
    def classify_questions(self, questions: List[str], language: str) -> List[str]:
        """Classify a list of questions."""
        if self.llm is None:
            self.initialize_model()
        
        use_deep_thinking = "Qwen3" in self.model_name
        sampling_params = self.get_sampling_params()
        
        # Create prompts
        prompts = [self.create_classification_prompt(q, use_deep_thinking) for q in questions]
        
        # Generate classifications
        logger.info(f"Classifying {len(questions)} questions in {language}")
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Extract answers
        answers = []
        for output in outputs:
            text = output.outputs[0].text
            # Extract classification from response
            if "Classification:" in text:
                classification = text.split("Classification:")[-1].strip().split()[0]
                answers.append(classification)
            else:
                answers.append("Unknown")
                logger.warning(f"No classification found in response: {text[:100]}...")
        
        return answers


class MultilingualEvaluator:
    """Main evaluator class."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.languages = config.get('languages', 
            ['Russian', 'English', 'French', 'German', 'Hebrew', 'Arabic', 'Chinese'])
    
    def translate_dataset(self, df: pd.DataFrame, output_path: str) -> pd.DataFrame:
        """Translate English questions to other languages."""
        api_key = self.config.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment")
        
        translator = Translator(api_key)
        english_questions = df['English'].tolist()
        
        logger.info(f"Translating {len(english_questions)} questions...")
        translations = translator.translate_batch(english_questions)
        
        # Add translations to dataframe
        for lang in translator.target_languages:
            df[lang] = [t.get(lang, "") for t in translations]
        
        # Save translated dataset
        df.to_csv(output_path, index=False)
        logger.info(f"Saved translated dataset to {output_path}")
        
        return df
    
    def evaluate_models(self, df: pd.DataFrame, models: List[str], output_path: str) -> pd.DataFrame:
        """Evaluate multiple models on the dataset."""
        results_df = df.copy()
        
        for model_name in models:
            logger.info(f"Evaluating model: {model_name}")
            model_short_name = model_name.split('/')[-1]
            
            classifier = Classifier(model_name, self.config.get('tensor_parallel_size', 2))
            
            for language in self.languages:
                if language not in df.columns:
                    logger.warning(f"Language {language} not found in dataset")
                    continue
                
                questions = df[language].tolist()
                classifications = classifier.classify_questions(questions, language)
                
                # Add results to dataframe
                column_name = f'answer_{model_short_name}_{language}'
                results_df[column_name] = classifications
                
                # Calculate accuracy if labels are available
                if 'label' in df.columns:
                    correct = sum(1 for i, c in enumerate(classifications) 
                                if (c == "Mutable" and df.iloc[i]['label'] == 1) or 
                                   (c == "Immutable" and df.iloc[i]['label'] == 0))
                    accuracy = correct / len(classifications)
                    logger.info(f"{model_short_name} - {language}: Accuracy = {accuracy:.3f}")
        
        # Save results
        results_df.to_csv(output_path, index=False)
        logger.info(f"Saved evaluation results to {output_path}")
        
        return results_df
    
    def generate_report(self, results_df: pd.DataFrame, output_path: str):
        """Generate evaluation report."""
        report = {
            'models': {},
            'languages': {},
            'overall_results': {}
        }
        
        # Extract model names and languages from column names
        result_columns = [col for col in results_df.columns if col.startswith('answer_')]
        
        for col in result_columns:
            parts = col.split('_', 2)
            if len(parts) == 3:
                _, model, language = parts
                
                if model not in report['models']:
                    report['models'][model] = {}
                if language not in report['languages']:
                    report['languages'][language] = {}
                
                # Calculate metrics if labels are available
                if 'label' in results_df.columns:
                    predictions = results_df[col].map({'Mutable': 1, 'Immutable': 0})
                    labels = results_df['label']
                    
                    # Remove NaN values
                    mask = predictions.notna()
                    predictions = predictions[mask]
                    labels = labels[mask]
                    
                    if len(predictions) > 0:
                        accuracy = (predictions == labels).mean()
                        report['models'][model][language] = accuracy
                        report['languages'][language][model] = accuracy
        
        # Calculate overall metrics
        for model, langs in report['models'].items():
            if langs:
                report['overall_results'][model] = sum(langs.values()) / len(langs)
        
        # Save report
        with open(output_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        logger.info(f"Saved evaluation report to {output_path}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Generate and evaluate multilingual classifications")
    parser.add_argument('--mode', type=str, choices=['translate', 'classify', 'both'], 
                       default='classify', help='Operation mode')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--input_file', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output_file', type=str, help='Output CSV file')
    parser.add_argument('--models', nargs='+', help='Models to evaluate')
    parser.add_argument('--api_key', type=str, help='OpenAI API key')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Override config with command line arguments
    if args.api_key:
        config['openai_api_key'] = args.api_key
    
    # Set default output file if not provided
    if not args.output_file:
        input_path = Path(args.input_file)
        args.output_file = str(input_path.parent / f"{input_path.stem}_output{input_path.suffix}")
    
    # Load input data
    logger.info(f"Loading data from {args.input_file}")
    df = pd.read_csv(args.input_file)
    
    evaluator = MultilingualEvaluator(config)
    
    # Execute based on mode
    if args.mode in ['translate', 'both']:
        df = evaluator.translate_dataset(df, args.output_file.replace('.csv', '_translated.csv'))
    
    if args.mode in ['classify', 'both']:
        models = args.models or config.get('models', [
            "meta-llama/Llama-3.1-8B-Instruct",
            "google/gemma-2-9b-it"
        ])
        
        results_df = evaluator.evaluate_models(df, models, args.output_file)
        
        # Generate report
        report_path = args.output_file.replace('.csv', '_report.yaml')
        evaluator.generate_report(results_df, report_path)


if __name__ == "__main__":
    main()
