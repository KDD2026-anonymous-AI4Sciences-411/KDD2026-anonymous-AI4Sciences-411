"""
Performance Buffer for SE-RL Framework
====================================

This module manages performance history and metrics for the SE-RL framework.

Author: AI Research Engineer
Date: 2024
"""

import time
import logging
from typing import Dict, List, Optional, Any
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

class PerformanceBuffer:
    """Buffer for storing and managing performance history"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.best_performance = None
        self.best_iteration = -1
    
    def add_performance(self, iteration: int, metrics: Dict[str, float], 
                       prompt: str, code: str, agent_type: str = "single"):
        """Add performance metrics to buffer"""
        entry = {
            'iteration': iteration,
            'metrics': metrics,
            'prompt': prompt,
            'code': code,
            'agent_type': agent_type,
            'timestamp': time.time()
        }
        
        self.buffer.append(entry)
        
        # Update best performance
        if self.best_performance is None or metrics.get('PA', 0) > self.best_performance.get('PA', 0):
            self.best_performance = metrics
            self.best_iteration = iteration
    
    def get_recent_performance(self, n: int) -> List[Dict[str, Any]]:
        """Get the n most recent performance entries"""
        return list(self.buffer)[-n:]
    
    def get_best_performance(self) -> Optional[Dict[str, Any]]:
        """Get the best performance entry"""
        if self.best_performance is None:
            return None
        
        for entry in reversed(self.buffer):
            if entry['metrics'] == self.best_performance:
                return entry
        return None
    
    def get_performance_trend(self) -> Dict[str, List[float]]:
        """Get performance trend over iterations"""
        if not self.buffer:
            return {}
        
        iterations = [entry['iteration'] for entry in self.buffer]
        pas = [entry['metrics'].get('PA', 0) for entry in self.buffer]
        wrs = [entry['metrics'].get('WR', 0) for entry in self.buffer]
        glrs = [entry['metrics'].get('GLR', 0) for entry in self.buffer]
        
        return {
            'iterations': iterations,
            'PA': pas,
            'WR': wrs,
            'GLR': glrs
        }
    
    def get_performance_statistics(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.buffer:
            return {}
        
        pas = [entry['metrics'].get('PA', 0) for entry in self.buffer]
        wrs = [entry['metrics'].get('WR', 0) for entry in self.buffer]
        glrs = [entry['metrics'].get('GLR', 0) for entry in self.buffer]
        
        return {
            'mean_PA': np.mean(pas),
            'std_PA': np.std(pas),
            'max_PA': np.max(pas),
            'min_PA': np.min(pas),
            'mean_WR': np.mean(wrs),
            'std_WR': np.std(wrs),
            'mean_GLR': np.mean(glrs),
            'std_GLR': np.std(glrs)
        }
    
    def clear(self):
        """Clear the performance buffer"""
        self.buffer.clear()
        self.best_performance = None
        self.best_iteration = -1
    
    def save_to_file(self, filename: str):
        """Save performance buffer to file"""
        import json
        data = {
            'buffer': list(self.buffer),
            'best_performance': self.best_performance,
            'best_iteration': self.best_iteration
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filename: str):
        """Load performance buffer from file"""
        import json
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.buffer = deque(data['buffer'], maxlen=self.max_size)
        self.best_performance = data['best_performance']
        self.best_iteration = data['best_iteration'] 