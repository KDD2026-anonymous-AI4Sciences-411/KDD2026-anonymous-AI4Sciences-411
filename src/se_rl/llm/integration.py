import os
import json
import logging
import time
import requests
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import openai
from anthropic import Anthropic

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM integration"""
    provider: str = "openai"  # openai, anthropic, huggingface, local
    model_name: str = "gpt-4"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    timeout: int = 60
    retry_attempts: int = 3
    retry_delay: float = 1.0

class LLMIntegration:
    """Production-ready LLM integration class"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        self.tokenizer = None
        self.model = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client based on provider"""
        try:
            if self.config.provider == "openai":
                self._initialize_openai()
            elif self.config.provider == "anthropic":
                self._initialize_anthropic()
            elif self.config.provider == "huggingface":
                self._initialize_huggingface()
            elif self.config.provider == "local":
                self._initialize_local()
            else:
                raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
                
            logger.info(f"Successfully initialized {self.config.provider} LLM client")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {str(e)}")
            raise
    
    def _initialize_openai(self):
        """Initialize OpenAI client"""
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        if self.config.base_url:
            openai.api_base = self.config.base_url
        
        openai.api_key = api_key
        self.client = openai
    
    def _initialize_anthropic(self):
        """Initialize Anthropic client"""
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided")
        
        self.client = Anthropic(api_key=api_key)
    
    def _initialize_huggingface(self):
        """Initialize Hugging Face client"""
        api_key = self.config.api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError("Hugging Face API key not provided")
        
        self.client = {"api_key": api_key, "base_url": "xxx"}
    
    def _initialize_local(self):
        """Initialize local model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.info(f"Loaded local model: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to load local model: {str(e)}")
            raise
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using the configured LLM"""
        for attempt in range(self.config.retry_attempts):
            try:
                if self.config.provider == "openai":
                    return self._generate_openai(prompt, **kwargs)
                elif self.config.provider == "anthropic":
                    return self._generate_anthropic(prompt, **kwargs)
                elif self.config.provider == "huggingface":
                    return self._generate_huggingface(prompt, **kwargs)
                elif self.config.provider == "local":
                    return self._generate_local(prompt, **kwargs)
                else:
                    raise ValueError(f"Unsupported provider: {self.config.provider}")
                    
            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise
    
    def _generate_openai(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API"""
        try:
            response = self.client.ChatCompletion.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert AI Research Engineer specializing in reinforcement learning and quantitative finance."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                top_p=kwargs.get('top_p', self.config.top_p),
                timeout=kwargs.get('timeout', self.config.timeout)
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    def _generate_anthropic(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic API"""
        try:
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                top_p=kwargs.get('top_p', self.config.top_p),
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise
    
    def _generate_huggingface(self, prompt: str, **kwargs) -> str:
        """Generate text using Hugging Face API"""
        try:
            headers = {"Authorization": f"Bearer {self.client['api_key']}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                    "temperature": kwargs.get('temperature', self.config.temperature),
                    "top_p": kwargs.get('top_p', self.config.top_p),
                    "do_sample": True
                }
            }
            
            response = requests.post(
                f"{self.client['base_url']}/models/{self.config.model_name}",
                headers=headers,
                json=payload,
                timeout=kwargs.get('timeout', self.config.timeout)
            )
            response.raise_for_status()
            
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '').strip()
            else:
                return result.get('generated_text', '').strip()
                
        except Exception as e:
            logger.error(f"Hugging Face API error: {str(e)}")
            raise
    
    def _generate_local(self, prompt: str, **kwargs) -> str:
        """Generate text using local model"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                    temperature=kwargs.get('temperature', self.config.temperature),
                    top_p=kwargs.get('top_p', self.config.top_p),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the original prompt from the generated text
            return generated_text[len(prompt):].strip()
            
        except Exception as e:
            logger.error(f"Local model generation error: {str(e)}")
            raise
    
    def extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response"""
        # Look for code blocks
        code_blocks = []
        
        # Pattern 1: ```python ... ```
        import re
        python_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        code_blocks.extend(python_blocks)
        
        # Pattern 2: ``` ... ``` (without language specification)
        generic_blocks = re.findall(r'```\n(.*?)\n```', response, re.DOTALL)
        code_blocks.extend(generic_blocks)
        
        # Pattern 3: Code without markdown blocks
        if not code_blocks:
            # Look for function or class definitions
            lines = response.split('\n')
            code_lines = []
            in_code = False
            
            for line in lines:
                if any(keyword in line for keyword in ['def ', 'class ', 'import ', 'from ']):
                    in_code = True
                if in_code:
                    code_lines.append(line)
                if in_code and line.strip() == '' and len(code_lines) > 0:
                    # Check if we've reached the end of code
                    next_lines = lines[lines.index(line) + 1:lines.index(line) + 3]
                    if not any(keyword in ' '.join(next_lines) for keyword in ['def ', 'class ', 'import ', 'from ']):
                        break
            
            if code_lines:
                code_blocks.append('\n'.join(code_lines))
        
        # Return the first code block found, or the entire response if no code blocks
        if code_blocks:
            return code_blocks[0].strip()
        else:
            return response.strip()
    
    def generate_code(self, prompt: str, **kwargs) -> str:
        """Generate Python code using LLM"""
        response = self.generate_text(prompt, **kwargs)
        return self.extract_code_from_response(response)

class LLMFactory:
    """Factory class for creating LLM integration instances"""
    
    @staticmethod
    def create_llm(config_dict: Dict[str, Any]) -> LLMIntegration:
        """Create LLM integration instance from configuration"""
        config = LLMConfig(**config_dict)
        return LLMIntegration(config)
    
    @staticmethod
    def create_openai_llm(model_name: str = "gpt-4", api_key: Optional[str] = None) -> LLMIntegration:
        """Create OpenAI LLM integration"""
        config = LLMConfig(
            provider="openai",
            model_name=model_name,
            api_key=api_key
        )
        return LLMIntegration(config)
    
    @staticmethod
    def create_anthropic_llm(model_name: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None) -> LLMIntegration:
        """Create Anthropic LLM integration"""
        config = LLMConfig(
            provider="anthropic",
            model_name=model_name,
            api_key=api_key
        )
        return LLMIntegration(config)
    
    @staticmethod
    def create_huggingface_llm(model_name: str = "meta-llama/Llama-2-7b-hf", api_key: Optional[str] = None) -> LLMIntegration:
        """Create Hugging Face LLM integration"""
        config = LLMConfig(
            provider="huggingface",
            model_name=model_name,
            api_key=api_key
        )
        return LLMIntegration(config)
    
    @staticmethod
    def create_local_llm(model_name: str = "meta-llama/Llama-2-7b-hf") -> LLMIntegration:
        """Create local LLM integration"""
        config = LLMConfig(
            provider="local",
            model_name=model_name
        )
        return LLMIntegration(config) 