"""
Instruction Population for SE-RL Framework
========================================

This module manages population of instructions for LLM enhancement.

Author: AI Research Engineer
Date: 2024
"""

import logging
import random
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class InstructionPopulation:
    """Manages population of instructions for LLM enhancement"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.instructions = []
        self.performance_scores = []
        self.instruction_types = []  # Track instruction types
        self.creation_timestamps = []  # Track when instructions were created
    
    def add_instruction(self, instruction: str, performance: float, instruction_type: str = "general"):
        """Add instruction with associated performance"""
        if len(self.instructions) >= self.max_size:
            # Remove worst performing instruction
            worst_idx = np.argmin(self.performance_scores)
            self.instructions.pop(worst_idx)
            self.performance_scores.pop(worst_idx)
            self.instruction_types.pop(worst_idx)
            self.creation_timestamps.pop(worst_idx)
        
        self.instructions.append(instruction)
        self.performance_scores.append(performance)
        self.instruction_types.append(instruction_type)
        self.creation_timestamps.append(len(self.instructions))  # Simple counter
    
    def sample_historical_instructions(self, n: int, instruction_type: Optional[str] = None) -> List[str]:
        """Sample n historical instructions based on performance"""
        if not self.instructions:
            return []
        
        # Filter by instruction type if specified
        if instruction_type:
            filtered_indices = [i for i, it in enumerate(self.instruction_types) if it == instruction_type]
            if not filtered_indices:
                return []
            
            filtered_instructions = [self.instructions[i] for i in filtered_indices]
            filtered_scores = [self.performance_scores[i] for i in filtered_indices]
        else:
            filtered_instructions = self.instructions
            filtered_scores = self.performance_scores
        
        # Weighted sampling based on performance
        weights = np.array(filtered_scores) + 1e-6  # Avoid zero weights
        weights = weights / np.sum(weights)
        
        if n >= len(filtered_instructions):
            return filtered_instructions.copy()
        
        sampled_indices = np.random.choice(len(filtered_instructions), n, p=weights, replace=False)
        return [filtered_instructions[i] for i in sampled_indices]
    
    def get_best_instruction(self, instruction_type: Optional[str] = None) -> Optional[str]:
        """Get the best performing instruction"""
        if not self.instructions:
            return None
        
        # Filter by instruction type if specified
        if instruction_type:
            filtered_indices = [i for i, it in enumerate(self.instruction_types) if it == instruction_type]
            if not filtered_indices:
                return None
            
            filtered_scores = [self.performance_scores[i] for i in filtered_indices]
            best_idx = filtered_indices[np.argmax(filtered_scores)]
        else:
            best_idx = np.argmax(self.performance_scores)
        
        return self.instructions[best_idx]
    
    def update_instruction_performance(self, instruction: str, performance: float):
        """Update performance for existing instruction"""
        try:
            idx = self.instructions.index(instruction)
            self.performance_scores[idx] = performance
        except ValueError:
            # Instruction not found, add it
            self.add_instruction(instruction, performance)
    
    def get_instruction_statistics(self) -> dict:
        """Get statistics about the instruction population"""
        if not self.instructions:
            return {}
        
        return {
            'total_instructions': len(self.instructions),
            'mean_performance': np.mean(self.performance_scores),
            'std_performance': np.std(self.performance_scores),
            'max_performance': np.max(self.performance_scores),
            'min_performance': np.min(self.performance_scores),
            'instruction_types': list(set(self.instruction_types)),
            'type_counts': {it: self.instruction_types.count(it) for it in set(self.instruction_types)}
        }
    
    def get_top_instructions(self, n: int = 10) -> List[tuple]:
        """Get top n performing instructions"""
        if not self.instructions:
            return []
        
        # Sort by performance
        sorted_indices = np.argsort(self.performance_scores)[::-1]
        top_instructions = []
        
        for i in range(min(n, len(sorted_indices))):
            idx = sorted_indices[i]
            top_instructions.append((
                self.instructions[idx],
                self.performance_scores[idx],
                self.instruction_types[idx]
            ))
        
        return top_instructions
    
    def crossover_instructions(self, instruction1: str, instruction2: str) -> str:
        """Perform crossover between two instructions"""
        # Simple crossover: combine parts of both instructions
        lines1 = instruction1.split('\n')
        lines2 = instruction2.split('\n')
        
        # Take first half from instruction1, second half from instruction2
        mid_point = min(len(lines1), len(lines2)) // 2
        combined_lines = lines1[:mid_point] + lines2[mid_point:]
        
        return '\n'.join(combined_lines)
    
    def mutate_instruction(self, instruction: str, mutation_rate: float = 0.1) -> str:
        """Mutate an instruction with random changes"""
        lines = instruction.split('\n')
        mutated_lines = []
        
        for line in lines:
            if random.random() < mutation_rate:
                # Simple mutation: add or remove words
                words = line.split()
                if words and random.random() < 0.5:
                    # Remove a random word
                    if len(words) > 1:
                        words.pop(random.randint(0, len(words) - 1))
                else:
                    # Add a random word
                    mutation_words = ['optimize', 'enhance', 'improve', 'focus', 'consider']
                    words.append(random.choice(mutation_words))
                
                line = ' '.join(words)
            
            mutated_lines.append(line)
        
        return '\n'.join(mutated_lines)
    
    def evolve_instructions(self, evolution_rate: float = 0.2):
        """Evolve the instruction population"""
        if len(self.instructions) < 2:
            return
        
        # Select parents for crossover
        parent1 = self.get_best_instruction()
        parent2 = random.choice(self.instructions)
        
        # Perform crossover
        child = self.crossover_instructions(parent1, parent2)
        
        # Mutate child
        child = self.mutate_instruction(child)
        
        # Add child to population
        self.add_instruction(child, 0.0, "evolved")
    
    def clear(self):
        """Clear the instruction population"""
        self.instructions.clear()
        self.performance_scores.clear()
        self.instruction_types.clear()
        self.creation_timestamps.clear()
    
    def save_to_file(self, filename: str):
        """Save instruction population to file"""
        import json
        data = {
            'instructions': self.instructions,
            'performance_scores': self.performance_scores,
            'instruction_types': self.instruction_types,
            'creation_timestamps': self.creation_timestamps
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filename: str):
        """Load instruction population from file"""
        import json
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.instructions = data['instructions']
        self.performance_scores = data['performance_scores']
        self.instruction_types = data['instruction_types']
        self.creation_timestamps = data['creation_timestamps'] 