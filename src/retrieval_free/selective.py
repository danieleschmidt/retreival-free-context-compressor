"""Selective compression with content-aware ratios."""

import re
from typing import Dict, List, Tuple

from .core import CompressorBase, CompressionResult, ContextCompressor
from .observability import monitor_performance


class SelectiveCompressor(CompressorBase):
    """Content-aware compressor with different ratios for different content types."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        rules: Dict[str, float] = None
    ):
        super().__init__(model_name)
        
        # Default compression rules
        self.rules = rules or {
            "code": 4.0,           # Code needs less compression
            "legal": 4.0,          # Legal text needs precision
            "technical": 6.0,      # Technical content moderate compression
            "general": 8.0,        # General text standard compression
            "repetitive": 16.0,    # Highly repetitive content aggressive compression
            "boilerplate": 20.0    # Boilerplate can be heavily compressed
        }
        
        # Content detection patterns
        self.patterns = {
            "code": [
                r'def\s+\w+\s*\(',      # Python function definitions
                r'class\s+\w+\s*:',     # Python class definitions
                r'import\s+\w+',        # Import statements
                r'#include\s*<',        # C/C++ includes
                r'function\s+\w+\s*\(', # JavaScript functions
                r'public\s+class\s+',   # Java class definitions
                r'\{\s*\n.*\n\s*\}',    # Code blocks
            ],
            "legal": [
                r'pursuant to',
                r'notwithstanding',
                r'whereas',
                r'hereafter',
                r'heretofore',
                r'section\s+\d+',
                r'article\s+[IVX]+',
            ],
            "technical": [
                r'algorithm',
                r'protocol',
                r'specification',
                r'implementation',
                r'architecture',
                r'API\s+endpoint',
                r'configuration',
            ],
            "repetitive": [
                r'(.{10,}?)\1{3,}',     # Repeated patterns
                r'(\w+\s+){10,}',       # Very long word repetitions
            ],
            "boilerplate": [
                r'copyright\s+\d{4}',
                r'all rights reserved',
                r'terms of service',
                r'privacy policy',
                r'disclaimer',
                r'generated automatically',
            ]
        }
    
    @monitor_performance
    def compress(self, text: str, **kwargs) -> CompressionResult:
        """Compress text using content-aware selective compression."""
        # Step 1: Analyze content and split into sections
        sections = self._analyze_and_segment(text)
        
        # Step 2: Compress each section with appropriate ratio
        all_mega_tokens = []
        total_original_length = 0
        total_processing_time = 0
        
        for section_text, content_type, compression_ratio in sections:
            compressor = ContextCompressor(
                model_name=self.model_name,
                compression_ratio=compression_ratio
            )
            
            section_result = compressor.compress(section_text)
            
            # Add content type to metadata
            for token in section_result.mega_tokens:
                token.metadata["content_type"] = content_type
                token.metadata["section_compression_ratio"] = compression_ratio
            
            all_mega_tokens.extend(section_result.mega_tokens)
            total_original_length += section_result.original_length
            total_processing_time += section_result.processing_time
        
        # Step 3: Calculate overall metrics
        compressed_length = len(all_mega_tokens)
        overall_ratio = self.get_compression_ratio(total_original_length, compressed_length)
        
        return CompressionResult(
            mega_tokens=all_mega_tokens,
            original_length=total_original_length,
            compressed_length=compressed_length,
            compression_ratio=overall_ratio,
            processing_time=total_processing_time,
            metadata={
                "selective_compression": True,
                "sections_processed": len(sections),
                "content_types": [ct for _, ct, _ in sections],
                "compression_ratios": [cr for _, _, cr in sections],
                "model": self.model_name
            }
        )
    
    def decompress(self, mega_tokens: List, **kwargs) -> str:
        """Decompress mega-tokens back to text."""
        if not mega_tokens:
            return ""
        
        # Group tokens by content type and reconstruct
        sections_by_type = {}
        for token in mega_tokens:
            content_type = token.metadata.get("content_type", "general")
            if content_type not in sections_by_type:
                sections_by_type[content_type] = []
            sections_by_type[content_type].append(token)
        
        # Reconstruct each section
        reconstructed_parts = []
        for content_type, tokens in sections_by_type.items():
            section_text = []
            for token in tokens:
                if "source_text" in token.metadata:
                    section_text.append(token.metadata["source_text"])
            
            if section_text:
                reconstructed_parts.append(f"[{content_type.upper()}]\n" + " ".join(section_text))
        
        return "\n\n".join(reconstructed_parts)
    
    def _analyze_and_segment(self, text: str) -> List[Tuple[str, str, float]]:
        """Analyze content and segment with appropriate compression ratios."""
        # Split text into paragraphs for analysis
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        sections = []
        current_section = ""
        current_type = "general"
        
        for paragraph in paragraphs:
            detected_type = self._detect_content_type(paragraph)
            
            # If content type changes, start a new section
            if detected_type != current_type and current_section:
                compression_ratio = self.rules.get(current_type, 8.0)
                sections.append((current_section.strip(), current_type, compression_ratio))
                current_section = paragraph
                current_type = detected_type
            else:
                current_section += "\n\n" + paragraph
                if current_type == "general":  # Update type if we were in general
                    current_type = detected_type
        
        # Add the last section
        if current_section:
            compression_ratio = self.rules.get(current_type, 8.0)
            sections.append((current_section.strip(), current_type, compression_ratio))
        
        # If no sections were created, treat entire text as one section
        if not sections:
            detected_type = self._detect_content_type(text)
            compression_ratio = self.rules.get(detected_type, 8.0)
            sections.append((text, detected_type, compression_ratio))
        
        return sections
    
    def _detect_content_type(self, text: str) -> str:
        """Detect the content type of a text segment."""
        text_lower = text.lower()
        
        # Count pattern matches for each content type
        type_scores = {}
        
        for content_type, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.MULTILINE | re.DOTALL))
                score += matches
            
            if score > 0:
                # Normalize by text length to handle different sizes
                type_scores[content_type] = score / max(len(text), 1) * 1000
        
        # Additional heuristics
        
        # Code detection heuristics
        if self._has_code_characteristics(text):
            type_scores["code"] = type_scores.get("code", 0) + 5
        
        # Repetitive content detection
        if self._is_repetitive(text):
            type_scores["repetitive"] = type_scores.get("repetitive", 0) + 10
        
        # Legal document detection
        if self._has_legal_characteristics(text):
            type_scores["legal"] = type_scores.get("legal", 0) + 3
        
        # Return the type with highest score, or "general" if no clear type
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        
        return "general"
    
    def _has_code_characteristics(self, text: str) -> bool:
        """Check if text has characteristics of code."""
        # Look for common code patterns
        code_indicators = [
            text.count('{') > 0 and text.count('}') > 0,  # Braces
            text.count('(') > text.count(' ') * 0.1,      # High parentheses ratio
            text.count(';') > 2,                          # Semicolons
            re.search(r'^\s{4,}', text, re.MULTILINE),    # Consistent indentation
            '=' in text and text.count('=') > text.count(' ') * 0.05,  # Assignments
        ]
        
        return sum(code_indicators) >= 2
    
    def _is_repetitive(self, text: str) -> bool:
        """Check if text is highly repetitive."""
        words = text.split()
        if len(words) < 10:
            return False
        
        # Check for repeated phrases
        phrases = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        unique_phrases = set(phrases)
        
        # If less than 50% unique phrases, consider repetitive
        return len(unique_phrases) / len(phrases) < 0.5
    
    def _has_legal_characteristics(self, text: str) -> bool:
        """Check if text has characteristics of legal documents."""
        legal_indicators = [
            len(re.findall(r'\b\w{15,}\b', text)) > 5,    # Many long words
            text.count(',') > len(text.split()) * 0.15,   # High comma density
            any(phrase in text.lower() for phrase in [
                'shall', 'thereof', 'herein', 'hereby', 'whereas'
            ]),
            re.search(r'\(\w+\)', text) is not None,      # Parenthetical references
        ]
        
        return sum(legal_indicators) >= 2
    
    def get_compression_stats(self, result: CompressionResult) -> Dict:
        """Get detailed compression statistics by content type."""
        if not result.metadata.get("selective_compression"):
            return {}
        
        content_types = result.metadata.get("content_types", [])
        compression_ratios = result.metadata.get("compression_ratios", [])
        
        stats = {
            "content_type_distribution": {},
            "average_compression_by_type": {},
            "total_sections": len(content_types)
        }
        
        # Count content types
        for content_type in content_types:
            stats["content_type_distribution"][content_type] = \
                stats["content_type_distribution"].get(content_type, 0) + 1
        
        # Average compression ratios by type
        type_ratios = {}
        for content_type, ratio in zip(content_types, compression_ratios):
            if content_type not in type_ratios:
                type_ratios[content_type] = []
            type_ratios[content_type].append(ratio)
        
        for content_type, ratios in type_ratios.items():
            stats["average_compression_by_type"][content_type] = sum(ratios) / len(ratios)
        
        return stats