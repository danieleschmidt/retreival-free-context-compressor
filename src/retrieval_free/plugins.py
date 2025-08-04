"""Framework integration plugins."""

from typing import Any, Dict, List, Optional, Union

from .core import AutoCompressor, CompressionResult, MegaToken


class CompressorPlugin:
    """Plugin wrapper for integrating compressors with various frameworks."""
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any = None,
        compressor: Union[str, Any] = "rfcc-base-8x",
        compression_threshold: int = 10000
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.compression_threshold = compression_threshold
        
        # Load compressor
        if isinstance(compressor, str):
            self.compressor = AutoCompressor.from_pretrained(compressor)
        else:
            self.compressor = compressor
    
    def generate(self, prompt: str, context: str = "", **kwargs) -> str:
        """Generate text with automatic compression of long contexts."""
        # Check if context needs compression
        if context and len(context) > self.compression_threshold:
            compression_result = self.compressor.compress(context)
            compressed_context = self._mega_tokens_to_text(compression_result.mega_tokens)
            full_prompt = f"{prompt}\n\nContext: {compressed_context}"
        else:
            full_prompt = f"{prompt}\n\nContext: {context}" if context else prompt
        
        # Use the wrapped model for generation
        if hasattr(self.model, 'generate'):
            # Handle transformers models
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, **kwargs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        elif hasattr(self.model, '__call__'):
            # Handle callable models (like OpenAI API)
            return self.model(full_prompt, **kwargs)
        else:
            raise ValueError("Model must have 'generate' method or be callable")
    
    def _mega_tokens_to_text(self, mega_tokens: List[MegaToken]) -> str:
        """Convert mega-tokens to text representation."""
        parts = []
        for token in mega_tokens:
            if "source_text" in token.metadata:
                parts.append(token.metadata["source_text"])
        return " ".join(parts)


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
        self.compressor = None
    
    def main(self):
        """Main CLI entry point."""
        import argparse
        import sys
        
        parser = argparse.ArgumentParser(description="Retrieval-Free Context Compressor")
        parser.add_argument("--model", default="rfcc-base-8x", help="Compressor model to use")
        parser.add_argument("--input", required=True, help="Input file path")
        parser.add_argument("--output", help="Output file path (optional)")
        parser.add_argument("--ratio", type=float, help="Compression ratio override")
        parser.add_argument("--stats", action="store_true", help="Show compression statistics")
        
        args = parser.parse_args()
        
        # Load compressor
        try:
            self.compressor = AutoCompressor.from_pretrained(args.model)
            if args.ratio:
                self.compressor.compression_ratio = args.ratio
        except Exception as e:
            print(f"Error loading compressor: {e}")
            sys.exit(1)
        
        # Read input
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading input file: {e}")
            sys.exit(1)
        
        # Compress
        try:
            result = self.compressor.compress(text)
            
            if args.stats:
                self._print_stats(result)
            
            # Output compressed representation
            if args.output:
                self._save_compressed(result, args.output)
                print(f"Compressed representation saved to {args.output}")
            else:
                self._print_compressed(result)
                
        except Exception as e:
            print(f"Error during compression: {e}")
            sys.exit(1)
    
    def _print_stats(self, result: CompressionResult):
        """Print compression statistics."""
        print(f"Compression Statistics:")
        print(f"  Original length: {result.original_length:,} tokens")
        print(f"  Compressed length: {result.compressed_length:,} mega-tokens")
        print(f"  Compression ratio: {result.compression_ratio:.1f}×")
        print(f"  Processing time: {result.processing_time:.2f}s")
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
            "mega_tokens": [
                {
                    "vector": token.vector.tolist(),
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