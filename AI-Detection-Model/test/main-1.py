import tenseal as ts #pip install tenseal
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
   "The quick brown fox jumps over the lazy dog.",
   "I watched the sunset over the ocean.",
   "Artificial intelligence is transforming industries.",
   "The library was quiet and smelled like old books.",
   "He dreamed of traveling to distant galaxies.",
   "Innovation drives progress in the tech world.",
   "The chef prepared a delicious meal for the guests.",
   "Climate change poses a significant threat to global biodiversity.",
   "The athlete trained rigorously for the upcoming marathon.",
   "Music has the power to evoke deep emotional responses."
]

embeddings = model.encode(sentences)
print(embeddings)

query_text = "artificial intelligence"
query_embedding = model.encode([query_text])
print(query_embedding)

context = ts.context(
           ts.SCHEME_TYPE.CKKS,
           poly_modulus_degree = 8192,
           coeff_mod_bit_sizes = [60, 40, 40, 60]
         )


context.generate_galois_keys()
context.global_scale = 2**40

secret_context = context.serialize(save_secret_key = True)

context.make_context_public()
public_context = context.serialize()

context = ts.context_from(secret_context)

enc_queryvec = ts.ckks_vector(context, query_embedding[0].tolist())

cosine_similarity_ranking = []
for i in range(len(sentences)):
    enc_sentence = ts.ckks_vector(context, embeddings[i].tolist())
    dot_product = enc_queryvec.dot(enc_sentence)
    cosine_similarity = dot_product.decrypt()[0]
    cosine_similarity_ranking.append({sentences[i]: abs(cosine_similarity)})

search_results = sorted(cosine_similarity_ranking, key=lambda x: list(x.values())[0], reverse=True)
for item in search_results:
    print(item)
    print()