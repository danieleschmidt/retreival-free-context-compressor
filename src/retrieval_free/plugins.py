"""Integration plugins for various frameworks."""

import logging
from typing import Optional, Dict, Any, List, Union
import torch

logger = logging.getLogger(__name__)


class CompressorPlugin:
    """Plugin for integrating compression with transformer models."""
    
    def __init__(
        self,
        model,
        tokenizer, 
        compressor: str = "rfcc-base-8x",
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
            from .core.auto_compressor import AutoCompressor
            self.compressor = AutoCompressor.from_pretrained(compressor)
        else:
            self.compressor = compressor
            
        logger.info(f"Initialized CompressorPlugin with {compressor}")
    
    def generate(
        self,
        input_text: str,
        max_new_tokens: int = 100,
        **kwargs
    ) -> str:
        """Generate text with automatic compression.
        
        Args:
            input_text: Input text (may be compressed automatically)
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Check if compression is needed
        input_length = len(self.tokenizer.encode(input_text))
        
        if self.auto_compress and input_length > self.compression_threshold:
            logger.info(f"Auto-compressing input ({input_length} tokens)")
            
            # Compress input
            compression_result = self.compressor.compress(input_text)
            
            # Convert mega-tokens to model input
            compressed_input = self._mega_tokens_to_input(compression_result.mega_tokens)
            
            logger.info(
                f"Compressed {input_length} -> {len(compressed_input)} tokens "
                f"({compression_result.compression_ratio:.1f}x ratio)"
            )
        else:
            compressed_input = input_text
        
        # Generate with the model
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
    
    def _mega_tokens_to_input(self, mega_tokens: List) -> str:
        """Convert mega-tokens to model input format.
        
        Args:
            mega_tokens: List of mega-tokens
            
        Returns:
            Formatted input string
        """
        # Simple conversion - in practice this would be more sophisticated
        token_summaries = []
        
        for i, token in enumerate(mega_tokens):
            summary = f"[COMPRESSED_SEGMENT_{i}]"
            token_summaries.append(summary)
        
        return " ".join(token_summaries)
    
    def compress_and_generate(
        self,
        context: str,
        query: str,
        max_new_tokens: int = 200,
        **kwargs
    ) -> str:
        """Compress context and generate response to query.
        
        Args:
            context: Long context to compress
            query: Query to answer
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Generated response
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
        
        return self.generate(full_input, max_new_tokens=max_new_tokens, **kwargs)


class LangChainIntegration:
    """Integration with LangChain framework."""
    
    def __init__(self, compressor_name: str = "rfcc-base-8x"):
        """Initialize LangChain integration.
        
        Args:
            compressor_name: Name of compressor to use
        """
        from .core.auto_compressor import AutoCompressor
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
                        # Simple compression representation
                        inputs["context"] = f"[COMPRESSED: {len(result.mega_tokens)} segments from {result.original_length} tokens]"
                    
                    return super()._call(inputs)
            
            return CompressionChain(
                compressor=self.compressor,
                threshold=compression_threshold,
                llm=llm,
                prompt=prompt
            )
            
        except ImportError:
            logger.error("LangChain not installed. Install with: pip install langchain")
            return None


class CLIInterface:
    """Command-line interface for the compressor."""
    
    def __init__(self):
        """Initialize CLI interface."""
        self.compressor = None
        
    def main(self):
        """Main CLI entry point."""
        import argparse
        
        parser = argparse.ArgumentParser(
            description="Retrieval-Free Context Compressor CLI"
        )
        
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Compress command
        compress_parser = subparsers.add_parser("compress", help="Compress text")
        compress_parser.add_argument("input", help="Input text file or string")
        compress_parser.add_argument("--model", default="rfcc-base-8x", help="Model to use")
        compress_parser.add_argument("--ratio", type=float, default=8.0, help="Compression ratio")
        compress_parser.add_argument("--output", help="Output file for compressed data")
        
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
        from .core.auto_compressor import AutoCompressor
        import os
        
        # Load compressor
        self.compressor = AutoCompressor.from_pretrained(args.model)
        
        # Get input text
        if os.path.isfile(args.input):
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = args.input
        
        # Compress
        print(f"Compressing with {args.model}...")
        result = self.compressor.compress(text)
        
        # Display results
        print(f"Original tokens: {result.original_length}")
        print(f"Compressed tokens: {result.compressed_length}")
        print(f"Compression ratio: {result.compression_ratio:.1f}x")
        print(f"Processing time: {result.processing_time:.2f}s")
        
        # Save output if requested
        if args.output:
            # In practice, this would save the mega-tokens in a proper format
            with open(args.output, 'w') as f:
                f.write(f"# Compressed with {args.model}\n")
                f.write(f"# Original length: {result.original_length}\n")
                f.write(f"# Compressed length: {result.compressed_length}\n")
                f.write(f"# Compression ratio: {result.compression_ratio:.1f}x\n")
                f.write(f"# Mega-tokens: {len(result.mega_tokens)}\n")
            print(f"Saved compression info to {args.output}")
    
    def _handle_list_models(self, args):
        """Handle list-models command."""
        from .core.auto_compressor import AutoCompressor
        
        models = AutoCompressor.list_available_models()
        
        print("Available models:")
        print("-" * 50)
        for name, description in models.items():
            print(f"{name:20} {description}")
    
    def _handle_benchmark(self, args):
        """Handle benchmark command."""
        print(f"Benchmarking {args.model}...")
        print("Benchmark functionality coming soon!")


def main():
    """CLI entry point."""
    cli = CLIInterface()
    cli.main()


if __name__ == "__main__":
    main()