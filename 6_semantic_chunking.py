from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Same Tesla text but structured to show semantic grouping
tesla_text = """Paragraph 1:
The Amazon rainforest is one of the most biodiverse ecosystems on Earth. It spans across nine countries and is home to over 3 million species. Scientists believe many more species are yet to be discovered in its dense foliage.

Paragraph 2:
Deforestation remains a major threat to the Amazon. Trees are cut down for logging, agriculture, and cattle grazing, disrupting habitats and accelerating climate change. Several conservation efforts are underway to protect the remaining forest.

Paragraph 3:
In contrast, urban green spaces provide localized benefits. Parks in cities improve air quality, reduce heat, and promote mental well-being. Urban planning increasingly includes green areas to counteract the effects of dense development."""

# Semantic Chunker - groups by meaning, not structure
semantic_splitter = SemanticChunker(
    embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
    breakpoint_threshold_amount=80
)

chunks = semantic_splitter.split_text(tesla_text)

print("SEMANTIC CHUNKING RESULTS:")
print("=" * 50)
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()