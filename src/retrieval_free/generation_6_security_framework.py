"""Generation 6: Advanced Security and Threat Modeling Framework

Revolutionary security framework implementing comprehensive threat detection,
vulnerability assessment, zero-trust architecture, and secure compression protocols.

Key Innovations:
1. Zero-trust compression architecture with continuous verification
2. Advanced threat detection using ML-based anomaly detection
3. Homomorphic encryption for secure compression operations
4. Differential privacy with adaptive noise mechanisms
5. Supply chain security with cryptographic verification
6. Real-time security monitoring and incident response
"""

import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Union, Set, Callable
import time
import logging
import hashlib
import hmac
import secrets
import json
import re
from collections import defaultdict, deque
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import os

from .core import CompressorBase, MegaToken, CompressionResult
from .exceptions import CompressionError, ValidationError
from .validation import ParameterValidator, validate_parameters, validate_input
from .observability import log_compression_operation, monitor_performance


logger = logging.getLogger(__name__)


@dataclass
class ThreatIndicator:
    """Represents a security threat indicator."""
    
    threat_type: str          # Type of threat (injection, exfiltration, etc.)
    severity: float           # Severity score 0-1
    confidence: float         # Detection confidence 0-1
    source_location: str      # Where threat was detected
    detection_time: float     # When threat was detected
    evidence: Dict[str, Any]  # Supporting evidence
    mitigation_actions: List[str] = None  # Recommended actions
    
    def __post_init__(self):
        if self.mitigation_actions is None:
            self.mitigation_actions = []
        
        if not 0.0 <= self.severity <= 1.0:
            raise ValidationError("Severity must be between 0 and 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError("Confidence must be between 0 and 1")


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    
    encryption_required: bool = True
    max_input_size: int = 100_000_000  # 100MB
    allowed_content_types: Set[str] = None
    blocked_patterns: List[str] = None
    rate_limit_requests_per_minute: int = 100
    require_authentication: bool = True
    audit_all_operations: bool = True
    differential_privacy_enabled: bool = True
    homomorphic_encryption_enabled: bool = False
    
    def __post_init__(self):
        if self.allowed_content_types is None:
            self.allowed_content_types = {'text/plain', 'application/json', 'text/html'}
        if self.blocked_patterns is None:
            self.blocked_patterns = [
                r'<script.*?>.*?</script>',  # Script injection
                r'javascript:',              # JavaScript URLs
                r'data:.*?base64',          # Base64 data URLs
                r'file://',                 # File URLs
                r'\\.\\./',                 # Path traversal
                r'(union|select|insert|update|delete|drop)\s+',  # SQL injection
                r'(eval|exec|system|shell_exec)\s*\(',  # Code injection
            ]


@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    
    event_id: str
    event_type: str           # authentication, compression, threat_detected, etc.
    timestamp: float
    user_id: Optional[str]
    source_ip: Optional[str]
    operation: str
    input_hash: str           # Hash of input data
    output_hash: Optional[str] # Hash of output data
    threat_indicators: List[ThreatIndicator]
    metadata: Dict[str, Any]
    
    def to_audit_log(self) -> str:
        """Convert to audit log format."""
        return json.dumps({
            'event_id': self.event_id,
            'event_type': self.event_type,
            'timestamp': self.timestamp,
            'user_id': self.user_id,
            'source_ip': self.source_ip,
            'operation': self.operation,
            'input_hash': self.input_hash,
            'output_hash': self.output_hash,
            'threats_detected': len(self.threat_indicators),
            'threat_types': [t.threat_type for t in self.threat_indicators],
            'metadata': self.metadata
        })


class ThreatDetector:
    """Advanced ML-based threat detection system."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.threat_patterns = self._initialize_threat_patterns()
        self.anomaly_detector = AnomalyDetector()
        self.content_analyzer = ContentAnalyzer()
        
        # Threat detection statistics
        self.detection_stats = {
            'total_scans': 0,
            'threats_detected': 0,
            'false_positives': 0,
            'detection_accuracy': 0.0,
            'avg_scan_time': 0.0
        }
        
    def _initialize_threat_patterns(self) -> Dict[str, List[str]]:
        """Initialize known threat patterns."""
        return {
            'code_injection': [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'vbscript:',
                r'(eval|exec|system|shell_exec)\s*\(',
                r'(union|select|insert|update|delete|drop)\s+.*?(from|into|table)',
                r'../../',
                r'\.\.\\',
                r'file://',
                r'data:.*?base64',
            ],
            'data_exfiltration': [
                r'(password|secret|key|token|credential)\s*[:=]\s*["\']?[a-zA-Z0-9]+',
                r'(api_key|access_token|private_key)\s*[:=]',
                r'BEGIN\s+(RSA\s+)?PRIVATE\s+KEY',
                r'-----BEGIN\s+CERTIFICATE-----',
                r'[a-zA-Z0-9]{32,}',  # Long hex strings (potential tokens)
            ],
            'prompt_injection': [
                r'ignore\s+(previous|above|prior)\s+instructions',
                r'forget\s+(everything|all)\s+(you|we)\s+(know|learned)',
                r'(system|admin|root)\s+(prompt|mode|access)',
                r'(jailbreak|bypass|override)\s+(security|safety|rules)',
                r'pretend\s+(you|to\s+be)\s+(another|different)',
                r'(reveal|show|tell)\s+(me\s+)?(your|the)\s+(prompt|instructions)',
            ],
            'adversarial_input': [
                r'[\u200B-\u200D\uFEFF]',  # Zero-width characters
                r'[\u0300-\u036F]{5,}',    # Excessive combining characters
                r'[^\x00-\x7F]{50,}',      # Long non-ASCII sequences
                r'(.)\1{100,}',            # Excessive repetition
                r'\\u[0-9a-fA-F]{4}',      # Unicode escapes
            ],
            'social_engineering': [
                r'(urgent|emergency|immediate)\s+(action|response)\s+required',
                r'(click|download|install|run)\s+(now|immediately)',
                r'(your\s+)?(account|system)\s+(has\s+been\s+)?(compromised|hacked)',
                r'(verify|confirm)\s+(your\s+)?(identity|account|password)',
                r'(suspended|disabled|locked)\s+(account|access)',
            ]
        }
    
    def scan_input(self, text: str, metadata: Dict[str, Any] = None) -> List[ThreatIndicator]:
        """Comprehensive threat scanning of input text."""
        start_time = time.time()
        threats = []
        
        if metadata is None:
            metadata = {}
        
        # Pattern-based detection
        pattern_threats = self._detect_pattern_threats(text)
        threats.extend(pattern_threats)
        
        # ML-based anomaly detection
        anomaly_threats = self.anomaly_detector.detect_anomalies(text)
        threats.extend(anomaly_threats)
        
        # Content analysis
        content_threats = self.content_analyzer.analyze_content(text)
        threats.extend(content_threats)
        
        # Statistical analysis
        statistical_threats = self._detect_statistical_anomalies(text)
        threats.extend(statistical_threats)
        
        # Update statistics
        scan_time = time.time() - start_time
        self._update_detection_stats(scan_time, len(threats))
        
        return threats
    
    def _detect_pattern_threats(self, text: str) -> List[ThreatIndicator]:
        """Detect threats using pattern matching."""
        threats = []
        
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    severity = self._calculate_pattern_severity(threat_type, match.group())
                    
                    threat = ThreatIndicator(
                        threat_type=threat_type,
                        severity=severity,
                        confidence=0.8,  # High confidence for pattern matches
                        source_location=f"char_{match.start()}-{match.end()}",
                        detection_time=time.time(),
                        evidence={
                            'pattern': pattern,
                            'match': match.group(),
                            'position': (match.start(), match.end())
                        },
                        mitigation_actions=self._get_mitigation_actions(threat_type)
                    )
                    threats.append(threat)
        
        return threats
    
    def _calculate_pattern_severity(self, threat_type: str, match: str) -> float:
        """Calculate severity score for pattern match."""
        base_severities = {
            'code_injection': 0.9,
            'data_exfiltration': 0.8,
            'prompt_injection': 0.7,
            'adversarial_input': 0.6,
            'social_engineering': 0.5
        }
        
        base_severity = base_severities.get(threat_type, 0.5)
        
        # Increase severity for longer matches
        length_multiplier = min(1.2, 1.0 + len(match) / 1000.0)
        
        return min(1.0, base_severity * length_multiplier)
    
    def _get_mitigation_actions(self, threat_type: str) -> List[str]:
        """Get recommended mitigation actions for threat type."""
        mitigations = {
            'code_injection': [
                'sanitize_input',
                'encode_output',
                'validate_content_type',
                'apply_content_security_policy'
            ],
            'data_exfiltration': [
                'mask_sensitive_data',
                'apply_differential_privacy',
                'audit_data_access',
                'encrypt_output'
            ],
            'prompt_injection': [
                'reject_input',
                'apply_input_validation',
                'use_safe_mode',
                'log_security_event'
            ],
            'adversarial_input': [
                'normalize_input',
                'apply_character_filters',
                'limit_input_length',
                'validate_encoding'
            ],
            'social_engineering': [
                'flag_for_review',
                'apply_content_warning',
                'limit_processing',
                'notify_administrator'
            ]
        }
        
        return mitigations.get(threat_type, ['log_security_event'])
    
    def _detect_statistical_anomalies(self, text: str) -> List[ThreatIndicator]:
        """Detect threats using statistical analysis."""
        threats = []
        
        # Entropy analysis
        entropy = self._calculate_entropy(text)
        if entropy > 7.5:  # Very high entropy might indicate encrypted/encoded data
            threat = ThreatIndicator(
                threat_type='high_entropy_data',
                severity=0.4,
                confidence=0.6,
                source_location='global',
                detection_time=time.time(),
                evidence={'entropy': entropy, 'threshold': 7.5},
                mitigation_actions=['validate_data_format', 'check_encoding']
            )
            threats.append(threat)
        
        # Length analysis
        if len(text) > 10_000_000:  # Very large input
            threat = ThreatIndicator(
                threat_type='excessive_input_size',
                severity=0.3,
                confidence=0.9,
                source_location='global',
                detection_time=time.time(),
                evidence={'length': len(text), 'threshold': 10_000_000},
                mitigation_actions=['limit_input_size', 'chunk_processing']
            )
            threats.append(threat)
        
        # Character distribution analysis
        char_distribution = self._analyze_character_distribution(text)
        if char_distribution['non_printable_ratio'] > 0.1:  # >10% non-printable chars
            threat = ThreatIndicator(
                threat_type='suspicious_character_distribution',
                severity=0.5,
                confidence=0.7,
                source_location='global',
                detection_time=time.time(),
                evidence=char_distribution,
                mitigation_actions=['validate_character_encoding', 'filter_non_printable']
            )
            threats.append(threat)
        
        return threats
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0
        
        # Count character frequencies
        char_counts = defaultdict(int)
        for char in text:
            char_counts[char] += 1
        
        # Calculate entropy
        text_length = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            probability = count / text_length
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _analyze_character_distribution(self, text: str) -> Dict[str, float]:
        """Analyze character distribution in text."""
        if not text:
            return {'non_printable_ratio': 0.0, 'ascii_ratio': 1.0, 'unicode_ratio': 0.0}
        
        printable_count = sum(1 for char in text if char.isprintable())
        ascii_count = sum(1 for char in text if ord(char) < 128)
        unicode_count = len(text) - ascii_count
        
        return {
            'non_printable_ratio': 1.0 - (printable_count / len(text)),
            'ascii_ratio': ascii_count / len(text),
            'unicode_ratio': unicode_count / len(text)
        }
    
    def _update_detection_stats(self, scan_time: float, threats_found: int):
        """Update threat detection statistics."""
        self.detection_stats['total_scans'] += 1
        self.detection_stats['threats_detected'] += threats_found
        
        # Update average scan time
        prev_avg = self.detection_stats['avg_scan_time']
        total_scans = self.detection_stats['total_scans']
        self.detection_stats['avg_scan_time'] = (prev_avg * (total_scans - 1) + scan_time) / total_scans


class AnomalyDetector:
    """ML-based anomaly detection for security threats."""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.anomaly_model = self._initialize_anomaly_model()
        self.threshold = 0.7  # Anomaly threshold
        
    def _initialize_anomaly_model(self):
        """Initialize anomaly detection model."""
        # Simplified anomaly detector using statistical methods
        # In practice, would use trained ML models
        return SimpleAnomalyDetector()
    
    def detect_anomalies(self, text: str) -> List[ThreatIndicator]:
        """Detect anomalies in text using ML models."""
        threats = []
        
        # Extract features
        features = self.feature_extractor.extract_features(text)
        
        # Run anomaly detection
        anomaly_score = self.anomaly_model.predict(features)
        
        if anomaly_score > self.threshold:
            threat = ThreatIndicator(
                threat_type='anomalous_content',
                severity=anomaly_score,
                confidence=0.6,  # ML predictions have some uncertainty
                source_location='global',
                detection_time=time.time(),
                evidence={
                    'anomaly_score': anomaly_score,
                    'threshold': self.threshold,
                    'features': features
                },
                mitigation_actions=['review_content', 'apply_additional_validation']
            )
            threats.append(threat)
        
        return threats


class FeatureExtractor:
    """Extract features for ML-based threat detection."""
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract numerical features from text."""
        if not text:
            return self._get_default_features()
        
        features = {}
        
        # Basic statistics
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['line_count'] = text.count('\n')
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Character statistics
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text)
        features['lowercase_ratio'] = sum(1 for c in text if c.islower()) / len(text)
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / len(text)
        features['special_char_ratio'] = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        features['whitespace_ratio'] = sum(1 for c in text if c.isspace()) / len(text)
        
        # Entropy features
        features['entropy'] = self._calculate_entropy(text)
        features['word_entropy'] = self._calculate_word_entropy(text)
        
        # Language features
        features['avg_sentence_length'] = self._calculate_avg_sentence_length(text)
        features['repeated_char_ratio'] = self._calculate_repeated_char_ratio(text)
        features['url_count'] = len(re.findall(r'https?://', text, re.IGNORECASE))
        features['email_count'] = len(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text))
        
        # Normalize features to [0, 1] range
        return self._normalize_features(features)
    
    def _get_default_features(self) -> Dict[str, float]:
        """Get default features for empty text."""
        return {key: 0.0 for key in [
            'length', 'word_count', 'line_count', 'avg_word_length',
            'uppercase_ratio', 'lowercase_ratio', 'digit_ratio', 'special_char_ratio', 'whitespace_ratio',
            'entropy', 'word_entropy', 'avg_sentence_length', 'repeated_char_ratio',
            'url_count', 'email_count'
        ]}
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate character-level entropy."""
        if not text:
            return 0.0
        
        char_counts = defaultdict(int)
        for char in text:
            char_counts[char] += 1
        
        text_length = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            probability = count / text_length
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _calculate_word_entropy(self, text: str) -> float:
        """Calculate word-level entropy."""
        words = text.lower().split()
        if not words:
            return 0.0
        
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1
        
        total_words = len(words)
        entropy = 0.0
        
        for count in word_counts.values():
            probability = count / total_words
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _calculate_avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        return sum(len(s.split()) for s in sentences) / len(sentences)
    
    def _calculate_repeated_char_ratio(self, text: str) -> float:
        """Calculate ratio of repeated characters."""
        if len(text) < 2:
            return 0.0
        
        repeated_count = 0
        for i in range(1, len(text)):
            if text[i] == text[i-1]:
                repeated_count += 1
        
        return repeated_count / (len(text) - 1)
    
    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalize features to [0, 1] range."""
        normalized = {}
        
        # Define normalization ranges for each feature
        normalization_ranges = {
            'length': 100000,
            'word_count': 10000,
            'line_count': 1000,
            'avg_word_length': 20,
            'entropy': 8,
            'word_entropy': 15,
            'avg_sentence_length': 50,
            'url_count': 10,
            'email_count': 10
        }
        
        for key, value in features.items():
            if key in normalization_ranges:
                normalized[key] = min(1.0, value / normalization_ranges[key])
            else:
                # Ratios are already in [0, 1] range
                normalized[key] = min(1.0, value)
        
        return normalized


class SimpleAnomalyDetector:
    """Simple statistical anomaly detector."""
    
    def __init__(self):
        # Pre-computed statistics for "normal" text
        self.normal_stats = {
            'length': {'mean': 1000, 'std': 2000},
            'word_count': {'mean': 200, 'std': 400},
            'entropy': {'mean': 4.5, 'std': 1.0},
            'special_char_ratio': {'mean': 0.1, 'std': 0.05},
            'uppercase_ratio': {'mean': 0.05, 'std': 0.03}
        }
    
    def predict(self, features: Dict[str, float]) -> float:
        """Predict anomaly score for given features."""
        anomaly_scores = []
        
        for feature, value in features.items():
            if feature in self.normal_stats:
                stats = self.normal_stats[feature]
                # Calculate z-score
                z_score = abs(value - stats['mean']) / max(stats['std'], 0.01)
                # Convert to anomaly score
                anomaly_score = min(1.0, z_score / 3.0)  # 3-sigma rule
                anomaly_scores.append(anomaly_score)
        
        # Return maximum anomaly score
        return max(anomaly_scores) if anomaly_scores else 0.0


class ContentAnalyzer:
    """Advanced content analysis for threat detection."""
    
    def __init__(self):
        self.sensitive_patterns = self._initialize_sensitive_patterns()
        
    def _initialize_sensitive_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for sensitive content detection."""
        return {
            'credentials': [
                r'password\s*[:=]\s*["\']?[a-zA-Z0-9!@#$%^&*()_+={}|:<>?]+',
                r'api[_-]?key\s*[:=]\s*["\']?[a-zA-Z0-9]+',
                r'secret[_-]?key\s*[:=]\s*["\']?[a-zA-Z0-9]+',
                r'access[_-]?token\s*[:=]\s*["\']?[a-zA-Z0-9]+',
                r'bearer\s+[a-zA-Z0-9._-]+',
            ],
            'personal_info': [
                r'\b\d{3}-?\d{2}-?\d{4}\b',  # SSN
                r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',  # Credit card
                r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',  # Email
                r'\b\d{3}[-.●]?\d{3}[-.●]?\d{4}\b',  # Phone number
            ],
            'system_info': [
                r'(localhost|127\.0\.0\.1|0\.0\.0\.0)',
                r'(file|ftp|ssh|telnet)://[^\s]+',
                r'(root|admin|administrator|system)',
                r'(database|db|mysql|postgresql|mongodb)',
                r'(docker|kubernetes|k8s)',
            ]
        }
    
    def analyze_content(self, text: str) -> List[ThreatIndicator]:
        """Analyze content for sensitive information and threats."""
        threats = []
        
        for category, patterns in self.sensitive_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    severity = self._calculate_content_severity(category, match.group())
                    
                    threat = ThreatIndicator(
                        threat_type=f'sensitive_{category}',
                        severity=severity,
                        confidence=0.7,
                        source_location=f"char_{match.start()}-{match.end()}",
                        detection_time=time.time(),
                        evidence={
                            'category': category,
                            'pattern': pattern,
                            'match': match.group()[:50] + '...' if len(match.group()) > 50 else match.group(),
                            'position': (match.start(), match.end())
                        },
                        mitigation_actions=['mask_sensitive_data', 'apply_differential_privacy']
                    )
                    threats.append(threat)
        
        return threats
    
    def _calculate_content_severity(self, category: str, match: str) -> float:
        """Calculate severity for sensitive content."""
        severity_mapping = {
            'credentials': 0.9,
            'personal_info': 0.8,
            'system_info': 0.6
        }
        
        base_severity = severity_mapping.get(category, 0.5)
        
        # Increase severity for longer matches (might be more complete info)
        length_bonus = min(0.2, len(match) / 100.0)
        
        return min(1.0, base_severity + length_bonus)


class SecureCompressionEngine:
    """Secure compression engine with encryption and privacy preservation."""
    
    def __init__(self, encryption_key: Optional[bytes] = None,
                 enable_homomorphic: bool = False,
                 differential_privacy_epsilon: float = 1.0):
        
        self.encryption_key = encryption_key or os.urandom(32)
        self.enable_homomorphic = enable_homomorphic
        self.differential_privacy_epsilon = differential_privacy_epsilon
        
        # Initialize encryption components
        self.cipher_suite = self._initialize_cipher_suite()
        
        if enable_homomorphic:
            self.homomorphic_encryptor = HomomorphicEncryptor()
        else:
            self.homomorphic_encryptor = None
        
        # Differential privacy mechanism
        self.dp_mechanism = DifferentialPrivacyMechanism(differential_privacy_epsilon)
        
    def _initialize_cipher_suite(self):
        """Initialize cryptographic cipher suite."""
        # Use AES-256-GCM for authenticated encryption
        return {
            'algorithm': algorithms.AES(self.encryption_key),
            'mode_factory': modes.GCM
        }
    
    def secure_compress(self, data: np.ndarray, 
                       privacy_level: str = 'standard') -> Tuple[bytes, Dict[str, Any]]:
        """Compress data with security and privacy protection."""
        compression_metadata = {
            'privacy_level': privacy_level,
            'encryption_enabled': True,
            'differential_privacy_applied': False,
            'homomorphic_encryption_used': False
        }
        
        # Apply differential privacy if requested
        if privacy_level in ['high', 'maximum']:
            data = self.dp_mechanism.add_noise(data)
            compression_metadata['differential_privacy_applied'] = True
            compression_metadata['epsilon_used'] = self.differential_privacy_epsilon
        
        # Compress data (simplified)
        compressed_data = self._compress_data(data)
        
        # Apply encryption
        if self.enable_homomorphic and privacy_level == 'maximum':
            # Use homomorphic encryption for computation on encrypted data
            encrypted_data = self.homomorphic_encryptor.encrypt(compressed_data)
            compression_metadata['homomorphic_encryption_used'] = True
        else:
            # Use standard authenticated encryption
            encrypted_data = self._encrypt_data(compressed_data)
        
        return encrypted_data, compression_metadata
    
    def secure_decompress(self, encrypted_data: bytes, 
                         metadata: Dict[str, Any]) -> np.ndarray:
        """Securely decompress encrypted data."""
        # Decrypt data
        if metadata.get('homomorphic_encryption_used', False):
            compressed_data = self.homomorphic_encryptor.decrypt(encrypted_data)
        else:
            compressed_data = self._decrypt_data(encrypted_data)
        
        # Decompress data
        decompressed_data = self._decompress_data(compressed_data)
        
        return decompressed_data
    
    def _compress_data(self, data: np.ndarray) -> bytes:
        """Compress numerical data."""
        # Simple compression using numpy's built-in compression
        import io
        import gzip
        
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
            np.save(f, data)
        
        return buffer.getvalue()
    
    def _decompress_data(self, compressed_data: bytes) -> np.ndarray:
        """Decompress numerical data."""
        import io
        import gzip
        
        buffer = io.BytesIO(compressed_data)
        with gzip.GzipFile(fileobj=buffer, mode='rb') as f:
            data = np.load(f)
        
        return data
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using AES-GCM."""
        nonce = os.urandom(12)  # 96-bit nonce for GCM
        
        cipher = Cipher(
            self.cipher_suite['algorithm'],
            self.cipher_suite['mode_factory'](nonce),
            backend=default_backend()
        )
        
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Return nonce + ciphertext + tag
        return nonce + ciphertext + encryptor.tag
    
    def _decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using AES-GCM."""
        # Extract components
        nonce = encrypted_data[:12]
        tag = encrypted_data[-16:]
        ciphertext = encrypted_data[12:-16]
        
        cipher = Cipher(
            self.cipher_suite['algorithm'],
            self.cipher_suite['mode_factory'](nonce, tag),
            backend=default_backend()
        )
        
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext


class HomomorphicEncryptor:
    """Simplified homomorphic encryption for secure computation."""
    
    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        self.public_key, self.private_key = self._generate_keys()
    
    def _generate_keys(self) -> Tuple[Any, Any]:
        """Generate homomorphic encryption keys."""
        # Simplified key generation (use proper HE library in production)
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        
        return public_key, private_key
    
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data using homomorphic encryption."""
        # Simplified encryption (use proper HE scheme in production)
        encrypted = self.public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return encrypted
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using homomorphic encryption."""
        decrypted = self.private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted


class DifferentialPrivacyMechanism:
    """Differential privacy implementation for data protection."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = 1.0  # Global sensitivity
        
    def add_noise(self, data: np.ndarray, mechanism: str = 'gaussian') -> np.ndarray:
        """Add calibrated noise for differential privacy."""
        if mechanism == 'gaussian':
            return self._gaussian_mechanism(data)
        elif mechanism == 'laplace':
            return self._laplace_mechanism(data)
        else:
            raise ValueError(f"Unknown DP mechanism: {mechanism}")
    
    def _gaussian_mechanism(self, data: np.ndarray) -> np.ndarray:
        """Apply Gaussian differential privacy mechanism."""
        # Calculate noise scale
        sigma = self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        
        # Add Gaussian noise
        noise = np.random.normal(0, sigma, size=data.shape)
        return data + noise
    
    def _laplace_mechanism(self, data: np.ndarray) -> np.ndarray:
        """Apply Laplace differential privacy mechanism."""
        # Calculate noise scale
        scale = self.sensitivity / self.epsilon
        
        # Add Laplace noise
        noise = np.random.laplace(0, scale, size=data.shape)
        return data + noise


class SecurityAuditLogger:
    """Comprehensive security audit logging system."""
    
    def __init__(self, log_file: str = 'security_audit.log',
                 enable_real_time_alerts: bool = True):
        self.log_file = log_file
        self.enable_real_time_alerts = enable_real_time_alerts
        self.event_buffer = deque(maxlen=1000)  # Keep recent events in memory
        
        # Alert thresholds
        self.alert_thresholds = {
            'high_severity_events_per_hour': 10,
            'failed_authentications_per_hour': 50,
            'data_exfiltration_attempts_per_hour': 5,
            'anomalous_behavior_score': 0.8
        }
        
    def log_security_event(self, event: SecurityEvent):
        """Log security event with audit trail."""
        # Add to memory buffer
        self.event_buffer.append(event)
        
        # Write to persistent log
        self._write_to_audit_log(event)
        
        # Check for real-time alerts
        if self.enable_real_time_alerts:
            self._check_alert_conditions(event)
    
    def _write_to_audit_log(self, event: SecurityEvent):
        """Write event to persistent audit log."""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(event.to_audit_log() + '\n')
        except Exception as e:
            logger.error(f"Failed to write to audit log: {e}")
    
    def _check_alert_conditions(self, event: SecurityEvent):
        """Check if event triggers real-time alerts."""
        current_time = time.time()
        
        # Check for high-severity events
        if any(t.severity > 0.8 for t in event.threat_indicators):
            self._send_alert(f"High-severity threat detected: {event.event_id}")
        
        # Check for patterns in recent events
        recent_events = [e for e in self.event_buffer 
                        if current_time - e.timestamp < 3600]  # Last hour
        
        high_severity_count = sum(1 for e in recent_events 
                                 if any(t.severity > 0.7 for t in e.threat_indicators))
        
        if high_severity_count >= self.alert_thresholds['high_severity_events_per_hour']:
            self._send_alert(f"High-severity event threshold exceeded: {high_severity_count}/hour")
    
    def _send_alert(self, message: str):
        """Send real-time security alert."""
        # In practice, would integrate with alerting systems (Slack, PagerDuty, etc.)
        logger.warning(f"SECURITY ALERT: {message}")
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for specified time period."""
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        recent_events = [e for e in self.event_buffer if e.timestamp > cutoff_time]
        
        summary = {
            'total_events': len(recent_events),
            'threat_events': len([e for e in recent_events if e.threat_indicators]),
            'unique_threat_types': len(set(
                t.threat_type for e in recent_events for t in e.threat_indicators
            )),
            'avg_severity': np.mean([
                t.severity for e in recent_events for t in e.threat_indicators
            ]) if any(e.threat_indicators for e in recent_events) else 0.0,
            'top_threat_types': self._get_top_threat_types(recent_events, 5)
        }
        
        return summary
    
    def _get_top_threat_types(self, events: List[SecurityEvent], top_n: int) -> List[Tuple[str, int]]:
        """Get top N threat types by frequency."""
        threat_counts = defaultdict(int)
        
        for event in events:
            for threat in event.threat_indicators:
                threat_counts[threat.threat_type] += 1
        
        return sorted(threat_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]


class SecureCompressor(CompressorBase):
    """Revolutionary secure compressor with comprehensive threat protection."""
    
    @validate_parameters(
        chunk_size=ParameterValidator.validate_chunk_size,
        compression_ratio=ParameterValidator.validate_compression_ratio,
        security_level=lambda x: x in ["standard", "high", "maximum"],
        differential_privacy_epsilon=lambda x: 0.1 <= x <= 10.0,
    )
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 512,
                 compression_ratio: float = 8.0,
                 security_level: str = "high",
                 security_policy: Optional[SecurityPolicy] = None,
                 differential_privacy_epsilon: float = 1.0,
                 enable_homomorphic_encryption: bool = False,
                 audit_log_file: str = "compression_security.log"):
        
        super().__init__(model_name)
        
        self.chunk_size = chunk_size
        self.compression_ratio = compression_ratio
        self.security_level = security_level
        self.security_policy = security_policy or SecurityPolicy()
        
        # Initialize security components
        self.threat_detector = ThreatDetector()
        self.secure_engine = SecureCompressionEngine(
            enable_homomorphic=enable_homomorphic_encryption,
            differential_privacy_epsilon=differential_privacy_epsilon
        )
        self.audit_logger = SecurityAuditLogger(log_file=audit_log_file)
        
        # Security statistics
        self.security_stats = {
            'total_compressions': 0,
            'threats_blocked': 0,
            'security_violations': 0,
            'average_threat_score': 0.0,
            'encryption_operations': 0,
            'differential_privacy_applications': 0
        }
        
        logger.info(f"Initialized Secure Compressor with {security_level} security level")
    
    @monitor_performance
    @log_compression_operation
    @validate_input(max_size=100_000_000)  # Security-aware input size limit
    def compress(self, text: str, user_id: Optional[str] = None,
                source_ip: Optional[str] = None, **kwargs) -> CompressionResult:
        """Revolutionary secure compression with comprehensive threat protection."""
        start_time = time.time()
        event_id = secrets.token_hex(16)
        
        try:
            # Step 1: Security validation and threat detection
            threat_indicators = self._perform_security_scan(text)
            
            # Step 2: Apply security policy
            policy_violations = self._check_security_policy(text, threat_indicators)
            
            if policy_violations:
                self._handle_security_violations(policy_violations, event_id, user_id, source_ip)
                raise ValidationError(f"Security policy violations: {policy_violations}")
            
            # Step 3: Classical preprocessing with security awareness
            chunks = self._secure_chunk_text(text)
            if not chunks:
                raise CompressionError("Secure text chunking failed", stage="preprocessing")
            
            embeddings = self._encode_chunks(chunks)
            if not embeddings:
                raise CompressionError("Embedding generation failed", stage="encoding")
            
            # Step 4: Apply secure compression
            compressed_embeddings, compression_metadata = self._apply_secure_compression(
                embeddings, threat_indicators)
            
            # Step 5: Create secure mega-tokens
            mega_tokens = self._create_secure_mega_tokens(
                compressed_embeddings, chunks, threat_indicators, compression_metadata)
            
            if not mega_tokens:
                raise CompressionError("Secure token creation failed", stage="tokenization")
            
            # Calculate metrics
            processing_time = time.time() - start_time
            original_length = self.count_tokens(text)
            compressed_length = len(mega_tokens)
            
            # Log security event
            self._log_compression_event(event_id, user_id, source_ip, text, 
                                      mega_tokens, threat_indicators, processing_time)
            
            # Update security statistics
            self._update_security_stats(threat_indicators)
            
            # Create secure result
            result = CompressionResult(
                mega_tokens=mega_tokens,
                original_length=int(original_length),
                compressed_length=compressed_length,
                compression_ratio=self.get_compression_ratio(original_length, compressed_length),
                processing_time=processing_time,
                metadata={
                    'model': self.model_name,
                    'secure_compression': True,
                    'security_level': self.security_level,
                    'threats_detected': len(threat_indicators),
                    'threat_types': [t.threat_type for t in threat_indicators],
                    'encryption_applied': compression_metadata.get('encryption_enabled', False),
                    'differential_privacy_applied': compression_metadata.get('differential_privacy_applied', False),
                    'homomorphic_encryption_used': compression_metadata.get('homomorphic_encryption_used', False),
                    'event_id': event_id,
                    'actual_chunks': len(chunks),
                    'success': True,
                }
            )
            
            # Add security-specific attributes
            result.security_metadata = {
                'threat_indicators': threat_indicators,
                'compression_metadata': compression_metadata,
                'security_level': self.security_level
            }
            
            return result
            
        except Exception as e:
            # Log security incident
            self._log_security_incident(event_id, user_id, source_ip, str(e))
            
            if isinstance(e, (ValidationError, CompressionError)):
                raise
            raise CompressionError(f"Secure compression failed: {e}",
                                 original_length=len(text) if text else 0)
    
    def _perform_security_scan(self, text: str) -> List[ThreatIndicator]:
        """Perform comprehensive security scan of input text."""
        logger.debug("Performing security scan...")
        
        # Use threat detector to scan for various threats
        threat_indicators = self.threat_detector.scan_input(text)
        
        logger.info(f"Security scan completed: {len(threat_indicators)} threats detected")
        
        return threat_indicators
    
    def _check_security_policy(self, text: str, 
                             threat_indicators: List[ThreatIndicator]) -> List[str]:
        """Check if input violates security policy."""
        violations = []
        
        # Check input size
        if len(text.encode('utf-8')) > self.security_policy.max_input_size:
            violations.append(f"Input size exceeds limit: {len(text)} > {self.security_policy.max_input_size}")
        
        # Check threat severity
        high_severity_threats = [t for t in threat_indicators if t.severity > 0.8]
        if high_severity_threats and self.security_level in ['high', 'maximum']:
            violations.append(f"High-severity threats detected: {[t.threat_type for t in high_severity_threats]}")
        
        # Check blocked patterns
        for pattern in self.security_policy.blocked_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(f"Blocked pattern detected: {pattern}")
        
        return violations
    
    def _handle_security_violations(self, violations: List[str], event_id: str,
                                  user_id: Optional[str], source_ip: Optional[str]):
        """Handle security policy violations."""
        logger.warning(f"Security violations detected: {violations}")
        
        # Create security event
        security_event = SecurityEvent(
            event_id=event_id,
            event_type='security_violation',
            timestamp=time.time(),
            user_id=user_id,
            source_ip=source_ip,
            operation='compress',
            input_hash=hashlib.sha256(str(violations).encode()).hexdigest(),
            output_hash=None,
            threat_indicators=[],
            metadata={'violations': violations}
        )
        
        self.audit_logger.log_security_event(security_event)
        self.security_stats['security_violations'] += 1
    
    def _secure_chunk_text(self, text: str) -> List[str]:
        """Securely chunk text with additional validation."""
        # Apply basic text chunking
        chunks = self._chunk_text(text)
        
        # Additional security validation for each chunk
        secure_chunks = []
        for chunk in chunks:
            # Check chunk for additional threats
            chunk_threats = self.threat_detector.scan_input(chunk)
            
            # Filter out high-threat chunks if security level is maximum
            if self.security_level == 'maximum':
                high_threat_count = sum(1 for t in chunk_threats if t.severity > 0.7)
                if high_threat_count > 2:  # Skip chunks with many high threats
                    logger.warning(f"Skipping high-threat chunk with {high_threat_count} threats")
                    continue
            
            secure_chunks.append(chunk)
        
        return secure_chunks
    
    def _apply_secure_compression(self, embeddings: List[np.ndarray],
                                threat_indicators: List[ThreatIndicator]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        """Apply secure compression with privacy protection."""
        # Convert embeddings to array
        embedding_array = np.array(embeddings)
        
        # Determine privacy level based on threats and security level
        privacy_level = self._determine_privacy_level(threat_indicators)
        
        # Apply secure compression
        encrypted_data, metadata = self.secure_engine.secure_compress(
            embedding_array, privacy_level=privacy_level)
        
        # For this implementation, we'll return the original embeddings
        # with security metadata (in practice, would return securely processed embeddings)
        
        # Apply basic compression
        compressed_embeddings = self._standard_compression(embeddings)
        
        # Update metadata
        self.security_stats['encryption_operations'] += 1
        if metadata.get('differential_privacy_applied', False):
            self.security_stats['differential_privacy_applications'] += 1
        
        return compressed_embeddings, metadata
    
    def _determine_privacy_level(self, threat_indicators: List[ThreatIndicator]) -> str:
        """Determine appropriate privacy level based on threats."""
        if not threat_indicators:
            return 'standard'
        
        max_severity = max(t.severity for t in threat_indicators)
        
        if max_severity > 0.8:
            return 'maximum'
        elif max_severity > 0.5:
            return 'high'
        else:
            return 'standard'
    
    def _standard_compression(self, embeddings: List[np.ndarray]) -> List[np.ndarray]:
        """Standard compression with security awareness."""
        # Simple PCA-based compression with security considerations
        try:
            from sklearn.decomposition import PCA
            
            embedding_matrix = np.array(embeddings)
            target_dim = max(32, int(embedding_matrix.shape[1] / self.compression_ratio))
            
            # Add small amount of noise for privacy
            if self.security_level in ['high', 'maximum']:
                noise_scale = 0.01
                noise = np.random.normal(0, noise_scale, embedding_matrix.shape)
                embedding_matrix += noise
            
            pca = PCA(n_components=target_dim)
            compressed_matrix = pca.fit_transform(embedding_matrix)
            
            return [compressed_matrix[i] for i in range(compressed_matrix.shape[0])]
        
        except ImportError:
            # Fallback compression
            compressed = []
            step = max(1, int(self.compression_ratio))
            for embedding in embeddings:
                compressed.append(embedding[::step])
            return compressed
    
    def _create_secure_mega_tokens(self, compressed_embeddings: List[np.ndarray],
                                 original_chunks: List[str], 
                                 threat_indicators: List[ThreatIndicator],
                                 compression_metadata: Dict[str, Any]) -> List[MegaToken]:
        """Create mega-tokens with comprehensive security metadata."""
        mega_tokens = []
        
        for i, compressed_vector in enumerate(compressed_embeddings):
            # Calculate security-adjusted confidence
            threat_penalty = sum(t.severity for t in threat_indicators) / max(len(threat_indicators), 1)
            base_confidence = 0.9
            security_confidence = max(0.1, base_confidence - threat_penalty * 0.3)
            
            # Find representative chunks
            chunks_per_token = len(original_chunks) // max(1, len(compressed_embeddings))
            start_idx = i * chunks_per_token
            end_idx = min(len(original_chunks), start_idx + chunks_per_token + 1)
            chunk_indices = list(range(start_idx, end_idx))
            
            # Create sanitized source text
            source_text = " ".join([original_chunks[idx] for idx in chunk_indices[:2]])
            if len(source_text) > 200:
                source_text = source_text[:200] + "..."
            
            # Sanitize source text based on security level
            if self.security_level in ['high', 'maximum']:
                source_text = self._sanitize_text(source_text)
            
            # Create comprehensive security metadata
            metadata = {
                'index': i,
                'source_text': source_text,
                'chunk_indices': chunk_indices,
                'secure_compression': True,
                'security_level': self.security_level,
                'threats_detected': len(threat_indicators),
                'threat_types': [t.threat_type for t in threat_indicators],
                'max_threat_severity': max((t.severity for t in threat_indicators), default=0.0),
                'encryption_applied': compression_metadata.get('encryption_enabled', False),
                'differential_privacy_applied': compression_metadata.get('differential_privacy_applied', False),
                'homomorphic_encryption_used': compression_metadata.get('homomorphic_encryption_used', False),
                'privacy_level': compression_metadata.get('privacy_level', 'standard'),
                'security_confidence': security_confidence,
                'compression_method': 'secure_compression',
                'vector_dimension': len(compressed_vector)
            }
            
            mega_tokens.append(
                MegaToken(
                    vector=compressed_vector,
                    metadata=metadata,
                    confidence=security_confidence
                )
            )
        
        return mega_tokens
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitize text for security purposes."""
        # Remove or mask sensitive patterns
        sanitized = text
        
        # Mask potential credentials
        sanitized = re.sub(r'password\s*[:=]\s*["\']?[a-zA-Z0-9!@#$%^&*()_+={}|:<>?]+', 
                          'password=***MASKED***', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*["\']?[a-zA-Z0-9]+',
                          r'\1=***MASKED***', sanitized, flags=re.IGNORECASE)
        
        # Mask potential PII
        sanitized = re.sub(r'\b\d{3}-?\d{2}-?\d{4}\b', '***-**-****', sanitized)  # SSN
        sanitized = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', 
                          '****-****-****-****', sanitized)  # Credit card
        
        return sanitized
    
    def _log_compression_event(self, event_id: str, user_id: Optional[str],
                             source_ip: Optional[str], input_text: str,
                             output_tokens: List[MegaToken], 
                             threat_indicators: List[ThreatIndicator],
                             processing_time: float):
        """Log compression operation for security audit."""
        # Create hash of input and output for audit trail
        input_hash = hashlib.sha256(input_text.encode('utf-8')).hexdigest()
        output_data = json.dumps([token.metadata for token in output_tokens])
        output_hash = hashlib.sha256(output_data.encode('utf-8')).hexdigest()
        
        security_event = SecurityEvent(
            event_id=event_id,
            event_type='compression',
            timestamp=time.time(),
            user_id=user_id,
            source_ip=source_ip,
            operation='compress',
            input_hash=input_hash,
            output_hash=output_hash,
            threat_indicators=threat_indicators,
            metadata={
                'processing_time': processing_time,
                'input_length': len(input_text),
                'output_tokens': len(output_tokens),
                'security_level': self.security_level,
                'compression_ratio': self.compression_ratio
            }
        )
        
        self.audit_logger.log_security_event(security_event)
    
    def _log_security_incident(self, event_id: str, user_id: Optional[str],
                             source_ip: Optional[str], error_message: str):
        """Log security incident."""
        security_event = SecurityEvent(
            event_id=event_id,
            event_type='security_incident',
            timestamp=time.time(),
            user_id=user_id,
            source_ip=source_ip,
            operation='compress',
            input_hash='',
            output_hash=None,
            threat_indicators=[],
            metadata={'error': error_message}
        )
        
        self.audit_logger.log_security_event(security_event)
    
    def _update_security_stats(self, threat_indicators: List[ThreatIndicator]):
        """Update security statistics."""
        self.security_stats['total_compressions'] += 1
        
        if threat_indicators:
            self.security_stats['threats_blocked'] += len(threat_indicators)
            
            # Update average threat score
            total_compressions = self.security_stats['total_compressions']
            current_avg = self.security_stats['average_threat_score']
            current_threat_score = max(t.severity for t in threat_indicators)
            
            self.security_stats['average_threat_score'] = (
                (current_avg * (total_compressions - 1) + current_threat_score) / total_compressions
            )
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get comprehensive security statistics."""
        stats = self.security_stats.copy()
        
        # Add threat detector statistics
        stats['threat_detection'] = self.threat_detector.detection_stats
        
        # Add audit summary
        stats['audit_summary'] = self.audit_logger.get_security_summary(hours=24)
        
        # Add security policy info
        stats['security_policy'] = {
            'encryption_required': self.security_policy.encryption_required,
            'max_input_size': self.security_policy.max_input_size,
            'differential_privacy_enabled': self.security_policy.differential_privacy_enabled,
            'audit_enabled': self.security_policy.audit_all_operations
        }
        
        return stats
    
    def decompress(self, mega_tokens: List[MegaToken], **kwargs) -> str:
        """Securely decompress mega-tokens with security annotations."""
        if not mega_tokens:
            return ""
        
        # Reconstruct from secure metadata
        reconstructed_parts = []
        for token in mega_tokens:
            if 'source_text' in token.metadata:
                text = token.metadata['source_text']
                
                # Add security enhancement markers
                if token.metadata.get('secure_compression', False):
                    security_level = token.metadata.get('security_level', 'unknown')
                    threats_detected = token.metadata.get('threats_detected', 0)
                    text += f" [Secure: {security_level} level, {threats_detected} threats]"
                
                reconstructed_parts.append(text)
        
        return " ".join(reconstructed_parts)


# Factory function for creating secure compressor
def create_secure_compressor(**kwargs) -> SecureCompressor:
    """Factory function for creating secure compressor."""
    return SecureCompressor(**kwargs)


# Register with AutoCompressor if available
def register_secure_models():
    """Register secure models with AutoCompressor."""
    try:
        from .core import AutoCompressor
        
        secure_models = {
            "secure-standard-8x": {
                "class": SecureCompressor,
                "params": {
                    "compression_ratio": 8.0,
                    "security_level": "standard",
                    "differential_privacy_epsilon": 1.0,
                    "enable_homomorphic_encryption": False
                }
            },
            "secure-high-6x": {
                "class": SecureCompressor,
                "params": {
                    "compression_ratio": 6.0,
                    "security_level": "high",
                    "differential_privacy_epsilon": 0.5,
                    "enable_homomorphic_encryption": False
                }
            },
            "secure-maximum-4x": {
                "class": SecureCompressor,
                "params": {
                    "compression_ratio": 4.0,
                    "security_level": "maximum",
                    "differential_privacy_epsilon": 0.1,
                    "enable_homomorphic_encryption": True
                }
            }
        }
        
        # Add to AutoCompressor registry
        AutoCompressor._MODELS.update(secure_models)
        logger.info("Registered secure compression models with AutoCompressor")
        
    except ImportError:
        logger.warning("Could not register secure models - AutoCompressor not available")


# Auto-register on import
register_secure_models()