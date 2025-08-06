"""Framework integration plugins."""

import logging
import json
from typing import Any, Dict, List, Optional, Union

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import with fallbacks for missing dependencies
try:
    from .core import AutoCompressor
    from .core.base import MegaToken, CompressionResult
    HAS_CORE = True
except ImportError:
    HAS_CORE = False
    # Mock classes for when dependencies are missing
    class MegaToken:
        def __init__(self, vector, metadata, confidence):
            self.vector = vector
            self.metadata = metadata
            self.confidence = confidence
    
    class CompressionResult:
        def __init__(self, mega_tokens, original_length, compressed_length, compression_ratio, processing_time, metadata):
            self.mega_tokens = mega_tokens
            self.original_length = original_length
            self.compressed_length = compressed_length 
            self.compression_ratio = compression_ratio
            self.processing_time = processing_time
            self.metadata = metadata

logger = logging.getLogger(__name__)


class CompressorPlugin:
    """Plugin wrapper for integrating compressors with various frameworks."""
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any = None,
        compressor: Union[str, Any] = "rfcc-base-8x",
        compression_threshold: int = 10000,
        auto_compress: bool = True
    ):
        """Initialize compressor plugin.
        
        Args:
            model: Base transformer model
            tokenizer: Model tokenizer
            compressor: Compressor model name or instance
            compression_threshold: Token threshold for auto-compression
            auto_compress: Whether to automatically compress long inputs
        """
        self.model = model
        self.tokenizer = tokenizer
        self.compression_threshold = compression_threshold
        self.auto_compress = auto_compress
        
        # Load compressor
        if isinstance(compressor, str):
            self.compressor = AutoCompressor.from_pretrained(compressor)
        else:
            self.compressor = compressor
            
        logger.info(f"Initialized CompressorPlugin with {compressor}")
    
    def generate(
        self,
        input_text: str,
        context: str = "",
        max_new_tokens: int = 100,
        **kwargs
    ) -> str:
        """Generate text with automatic compression.
        
        Args:
            input_text: Input text (may be compressed automatically)
            context: Optional context to compress
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Combine input and context if provided
        full_input = input_text
        if context:
            full_input = f"{input_text}\n\nContext: {context}"
        
        # Check if compression is needed
        if self.tokenizer:
            input_length = len(self.tokenizer.encode(full_input))
        else:
            input_length = len(full_input.split())
        
        if self.auto_compress and input_length > self.compression_threshold:
            logger.info(f"Auto-compressing input ({input_length} tokens)")
            
            # Compress input
            compression_result = self.compressor.compress(full_input)
            
            # Convert mega-tokens to model input
            compressed_input = self._mega_tokens_to_input(compression_result.mega_tokens)
            
            logger.info(
                f"Compressed {input_length} -> {len(compressed_input)} tokens "
                f"({compression_result.compression_ratio:.1f}x ratio)"
            )
        else:
            compressed_input = full_input
        
        # Generate with the model
        if hasattr(self.model, 'generate') and self.tokenizer:
            # Handle transformers models
            inputs = self.tokenizer(
                compressed_input,
                return_tensors="pt",
                truncation=True,
                max_length=self.model.config.max_position_embeddings
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    **kwargs
                )
            
            # Decode output
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input portion
            if compressed_input in generated:
                generated = generated.replace(compressed_input, "").strip()
            
            return generated
        elif hasattr(self.model, '__call__'):
            # Handle callable models (like OpenAI API)
            return self.model(compressed_input, **kwargs)
        else:
            raise ValueError("Model must have 'generate' method or be callable")
    
    def _mega_tokens_to_input(self, mega_tokens: List) -> str:
        """Convert mega-tokens to model input format.
        
        Args:
            mega_tokens: List of mega-tokens
            
        Returns:
            Formatted input string
        """
        # Check if mega-tokens have source_text metadata
        parts = []
        for i, token in enumerate(mega_tokens):
            if hasattr(token, 'metadata') and "source_text" in token.metadata:
                parts.append(token.metadata["source_text"])
            else:
                # Fallback to segment representation
                parts.append(f"[COMPRESSED_SEGMENT_{i}]")
        
        return " ".join(parts)
    
    def _mega_tokens_to_text(self, mega_tokens: List[MegaToken]) -> str:
        """Convert mega-tokens to text representation."""
        parts = []
        for token in mega_tokens:
            if "source_text" in token.metadata:
                parts.append(token.metadata["source_text"])
        return " ".join(parts)
    
    def compress_and_generate(
        self,
        context: str,
        query: str,
        max_new_tokens: int = 200,
        **kwargs
    ) -> Dict[str, Any]:
        """Compress context and generate response to query.
        
        Args:
            context: Long context to compress
            query: Query to answer
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated response or dict with metrics
        """
        # Compress context
        compression_result = self.compressor.compress(context)
        
        # Get relevant mega-tokens for the query
        if hasattr(self.compressor, 'get_attention_weights'):
            attention_weights = self.compressor.get_attention_weights(
                query, compression_result.mega_tokens
            )
            
            # Select top relevant tokens
            top_k = min(10, len(compression_result.mega_tokens))
            if len(attention_weights) > 0:
                top_indices = torch.topk(attention_weights, top_k).indices
                relevant_tokens = [compression_result.mega_tokens[i] for i in top_indices]
            else:
                relevant_tokens = compression_result.mega_tokens[:top_k]
        else:
            relevant_tokens = compression_result.mega_tokens
        
        # Format input with compressed context
        compressed_context = self._mega_tokens_to_input(relevant_tokens)
        full_input = f"Context: {compressed_context}\n\nQuestion: {query}\n\nAnswer:"
        
        generated_text = self.generate(full_input, max_new_tokens=max_new_tokens, **kwargs)
        
        return {
            "generated_text": generated_text,
            "compression_ratio": compression_result.compression_ratio,
            "original_context_length": compression_result.original_length,
            "compressed_context_length": compression_result.compressed_length,
            "processing_time": compression_result.processing_time,
        }


class HuggingFacePlugin(CompressorPlugin):
    """Specialized plugin for HuggingFace transformers."""
    
    def __init__(self, model_name: str, compressor: str = "rfcc-base-8x", **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        super().__init__(model, tokenizer, compressor)
        self.model_name = model_name
    
    def compress_and_generate(
        self,
        prompt: str,
        long_context: str,
        max_new_tokens: int = 200,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """Compress context and generate with detailed metrics."""
        # Compress the context
        compression_result = self.compressor.compress(long_context)
        
        # Convert to text for the model
        compressed_text = self._mega_tokens_to_text(compression_result.mega_tokens)
        
        # Generate
        full_prompt = f"{prompt}\n\nContext: {compressed_text}"
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **generation_kwargs
        )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return {
            "generated_text": generated_text,
            "compression_ratio": compression_result.compression_ratio,
            "original_context_length": compression_result.original_length,
            "compressed_context_length": compression_result.compressed_length,
            "processing_time": compression_result.processing_time,
            "prompt_tokens": inputs.input_ids.shape[1],
        }


class LangChainIntegration:
    """Integration with LangChain framework."""
    
    def __init__(self, compressor_name: str = "rfcc-base-8x"):
        """Initialize LangChain integration.
        
        Args:
            compressor_name: Name of compressor to use
        """
        self.compressor = AutoCompressor.from_pretrained(compressor_name)
        
    def create_compression_chain(
        self,
        llm,
        compression_threshold: int = 10000
    ):
        """Create a LangChain chain with compression.
        
        Args:
            llm: LangChain LLM instance
            compression_threshold: Token threshold for compression
            
        Returns:
            Compression chain
        """
        try:
            from langchain.chains import LLMChain
            from langchain.prompts import PromptTemplate
            from langchain.schema import BaseLanguageModel
            
            # Custom prompt template that handles compression
            template = """
            Context (may be compressed): {context}
            
            Question: {question}
            
            Answer based on the context provided:
            """
            
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=template
            )
            
            # Create chain with custom processing
            class CompressionChain(LLMChain):
                def __init__(self, compressor, threshold, **kwargs):
                    super().__init__(**kwargs)
                    self.compressor = compressor
                    self.threshold = threshold
                
                def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
                    # Compress context if needed
                    context = inputs.get("context", "")
                    
                    if len(context.split()) > self.threshold:
                        result = self.compressor.compress(context)
                        # Format compressed context
                        compressed_parts = []
                        for i, token in enumerate(result.mega_tokens):
                            if "source_text" in token.metadata:
                                confidence = token.confidence
                                text = token.metadata["source_text"]
                                compressed_parts.append(f"Section {i+1} (confidence: {confidence:.2f}): {text}")
                        
                        inputs["context"] = "\n\n".join(compressed_parts) if compressed_parts else f"[COMPRESSED: {len(result.mega_tokens)} segments from {result.original_length} tokens]"
                    
                    return super()._call(inputs)
                
                def run(self, document: str, question: str) -> Dict[str, Any]:
                    """Process long document and answer question."""
                    # Compress if needed
                    if len(document) > self.threshold:
                        compression_result = self.compressor.compress(document)
                        compressed_doc = self._format_compressed_context(compression_result)
                        
                        prompt_text = f"""Based on the following compressed document context, answer the question.

Compressed Context:
{compressed_doc}

Question: {question}

Answer:"""
                        
                        answer = self.llm(prompt_text)
                        
                        return {
                            "answer": answer,
                            "used_compression": True,
                            "compression_ratio": compression_result.compression_ratio,
                            "original_length": compression_result.original_length
                        }
                    else:
                        prompt_text = f"""Based on the following document, answer the question.

Document:
{document}

Question: {question}

Answer:"""
                        
                        answer = self.llm(prompt_text)
                        
                        return {
                            "answer": answer,
                            "used_compression": False,
                            "original_length": len(document)
                        }
                
                def _format_compressed_context(self, result: CompressionResult) -> str:
                    """Format compressed context for LLM consumption."""
                    sections = []
                    for i, token in enumerate(result.mega_tokens):
                        if "source_text" in token.metadata:
                            confidence = token.confidence
                            text = token.metadata["source_text"]
                            sections.append(f"Section {i+1} (confidence: {confidence:.2f}): {text}")
                    
                    return "\n\n".join(sections)
            
            return CompressionChain(
                compressor=self.compressor,
                threshold=compression_threshold,
                llm=llm,
                prompt=prompt
            )
            
        except ImportError:
            logger.error("LangChain not installed. Install with: pip install langchain")
            return None


try:
    import langchain
    from langchain.schema import BaseLanguageModel
    
    class CompressionChain:
        """LangChain integration for compressed context processing."""
        
        def __init__(
            self,
            llm: BaseLanguageModel,
            compressor: str = "rfcc-base-8x",
            compression_threshold: int = 10000
        ):
            self.llm = llm
            self.compressor = AutoCompressor.from_pretrained(compressor)
            self.compression_threshold = compression_threshold
        
        def run(self, document: str, question: str) -> Dict[str, Any]:
            """Process long document and answer question."""
            # Compress if needed
            if len(document) > self.compression_threshold:
                compression_result = self.compressor.compress(document)
                compressed_doc = self._format_compressed_context(compression_result)
                
                prompt = f"""Based on the following compressed document context, answer the question.

Compressed Context:
{compressed_doc}

Question: {question}

Answer:"""
                
                answer = self.llm(prompt)
                
                return {
                    "answer": answer,
                    "used_compression": True,
                    "compression_ratio": compression_result.compression_ratio,
                    "original_length": compression_result.original_length
                }
            else:
                prompt = f"""Based on the following document, answer the question.

Document:
{document}

Question: {question}

Answer:"""
                
                answer = self.llm(prompt)
                
                return {
                    "answer": answer,
                    "used_compression": False,
                    "original_length": len(document)
                }
        
        def _format_compressed_context(self, result: CompressionResult) -> str:
            """Format compressed context for LLM consumption."""
            sections = []
            for i, token in enumerate(result.mega_tokens):
                if "source_text" in token.metadata:
                    confidence = token.confidence
                    text = token.metadata["source_text"]
                    sections.append(f"Section {i+1} (confidence: {confidence:.2f}): {text}")
            
            return "\n\n".join(sections)

except ImportError:
    # LangChain not available
    class CompressionChain:
        def __init__(self, *args, **kwargs):
            raise ImportError("LangChain is required for CompressionChain. Install with: pip install langchain")


class OpenAIPlugin:
    """Plugin for OpenAI API integration."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        compressor: str = "rfcc-base-8x"
    ):
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model
            self.compressor = AutoCompressor.from_pretrained(compressor)
        except ImportError:
            raise ImportError("OpenAI library required. Install with: pip install openai")
    
    def chat_with_compression(
        self,
        messages: List[Dict],
        long_context: str = "",
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Chat with automatic context compression."""
        # Compress long context if provided
        if long_context:
            compression_result = self.compressor.compress(long_context)
            compressed_text = self._mega_tokens_to_text(compression_result.mega_tokens)
            
            # Add compressed context to the conversation
            context_message = {
                "role": "system",
                "content": f"Compressed context information: {compressed_text}"
            }
            messages = [context_message] + messages
        
        # Make API call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs
        )
        
        result = {
            "response": response.choices[0].message.content,
            "usage": response.usage._asdict() if response.usage else {},
        }
        
        if long_context:
            result.update({
                "compression_used": True,
                "compression_ratio": compression_result.compression_ratio,
                "original_context_length": compression_result.original_length,
            })
        
        return result
    
    def _mega_tokens_to_text(self, mega_tokens: List[MegaToken]) -> str:
        """Convert mega-tokens to text representation."""
        parts = []
        for token in mega_tokens:
            if "source_text" in token.metadata:
                parts.append(token.metadata["source_text"])
        return " ".join(parts)


class CLIInterface:
    """Command-line interface for the compressor."""
    
    def __init__(self):
        """Initialize CLI interface."""
        self.compressor = None
        
    def main(self):
        """Main CLI entry point."""
        import argparse
        import sys
        import os
        
        parser = argparse.ArgumentParser(
            description="Retrieval-Free Context Compressor CLI"
        )
        
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Compress command
        compress_parser = subparsers.add_parser("compress", help="Compress text")
        compress_parser.add_argument("input", help="Input text file or string")
        compress_parser.add_argument("--model", default="rfcc-base-8x", help="Model to use")
        compress_parser.add_argument("--ratio", type=float, help="Compression ratio override")
        compress_parser.add_argument("--output", help="Output file for compressed data")
        compress_parser.add_argument("--stats", action="store_true", help="Show compression statistics")
        
        # List models command
        list_parser = subparsers.add_parser("list-models", help="List available models")
        
        # Benchmark command
        benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
        benchmark_parser.add_argument("--model", default="rfcc-base-8x", help="Model to benchmark")
        benchmark_parser.add_argument("--dataset", help="Dataset to use for benchmarking")
        
        args = parser.parse_args()
        
        if args.command == "compress":
            self._handle_compress(args)
        elif args.command == "list-models":
            self._handle_list_models(args)
        elif args.command == "benchmark":
            self._handle_benchmark(args)
        else:
            parser.print_help()
    
    def _handle_compress(self, args):
        """Handle compress command."""
        import os
        import sys
        
        # Load compressor
        try:
            self.compressor = AutoCompressor.from_pretrained(args.model)
            if hasattr(args, 'ratio') and args.ratio:
                self.compressor.compression_ratio = args.ratio
        except Exception as e:
            print(f"Error loading compressor: {e}")
            sys.exit(1)
        
        # Get input text
        try:
            if os.path.isfile(args.input):
                with open(args.input, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                text = args.input
        except Exception as e:
            print(f"Error reading input: {e}")
            sys.exit(1)
        
        # Compress
        try:
            print(f"Compressing with {args.model}...")
            result = self.compressor.compress(text)
            
            # Display results
            if hasattr(args, 'stats') and args.stats:
                self._print_stats(result)
            else:
                print(f"Original tokens: {result.original_length}")
                print(f"Compressed tokens: {result.compressed_length}")
                print(f"Compression ratio: {result.compression_ratio:.1f}x")
                print(f"Processing time: {result.processing_time:.2f}s")
            
            # Save output if requested
            if args.output:
                self._save_compressed(result, args.output)
                print(f"Compressed representation saved to {args.output}")
            else:
                self._print_compressed(result)
                
        except Exception as e:
            print(f"Error during compression: {e}")
            sys.exit(1)
    
    def _handle_list_models(self, args):
        """Handle list-models command."""
        models = AutoCompressor.list_available_models()
        
        print("Available models:")
        print("-" * 50)
        for name, description in models.items():
            print(f"{name:20} {description}")
    
    def _handle_benchmark(self, args):
        """Handle benchmark command."""
        print(f"Benchmarking {args.model}...")
        if args.dataset:
            print(f"Using dataset: {args.dataset}")
        print("Benchmark functionality coming soon!")
    
    def _print_stats(self, result: CompressionResult):
        """Print compression statistics."""
        print(f"Compression Statistics:")
        print(f"  Original length: {result.original_length:,} tokens")
        print(f"  Compressed length: {result.compressed_length:,} mega-tokens")
        print(f"  Compression ratio: {result.compression_ratio:.1f}×")
        print(f"  Processing time: {result.processing_time:.2f}s")
        if hasattr(result, 'effective_compression'):
            print(f"  Effective compression: {result.effective_compression:.1f}×")
        print()
    
    def _print_compressed(self, result: CompressionResult):
        """Print compressed representation to stdout."""
        for i, token in enumerate(result.mega_tokens):
            print(f"Mega-Token {i+1}:")
            print(f"  Confidence: {token.confidence:.3f}")
            if "source_text" in token.metadata:
                preview = token.metadata["source_text"][:100]
                print(f"  Preview: {preview}...")
            print()
    
    def _save_compressed(self, result: CompressionResult, output_path: str):
        """Save compressed representation to file."""
        import json
        
        # Convert to serializable format
        data = {
            "model_info": {
                "compression_ratio": result.compression_ratio,
                "mega_tokens_count": len(result.mega_tokens)
            },
            "mega_tokens": [
                {
                    "vector": token.vector.tolist() if hasattr(token.vector, 'tolist') else token.vector,
                    "metadata": token.metadata,
                    "confidence": token.confidence
                }
                for token in result.mega_tokens
            ],
            "compression_result": {
                "original_length": result.original_length,
                "compressed_length": result.compressed_length,
                "compression_ratio": result.compression_ratio,
                "processing_time": result.processing_time,
                "metadata": result.metadata
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)


def main():
    """CLI entry point."""
    cli = CLIInterface()
    cli.main()


if __name__ == "__main__":
    main()
