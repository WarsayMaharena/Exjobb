import sys
sys.stdout.reconfigure(encoding="utf-8")

#AI modell byggd på CKKS schematics

import tenseal as ts # pip install tenseal
from deepface import DeepFace #!pip install deepface
import base64

# We are going to find vector representations of 
# facial images. This will be done in the client side.

#skapa kod som kan croppa ut ansikten från en bild.

img1_path = "dataset/img1.jpg"
img2_path = "dataset/img2.jpg"

img1_embedding = DeepFace.represent(
    img1_path,
    model_name="Facenet"
)[0]["embedding"]

img2_embedding = DeepFace.represent(
    img2_path,
    model_name="Facenet"
)[0]["embedding"]

def write_data(file_name, data):
    
    if type(data) == bytes:
        #bytes to base64
        data = base64.b64encode(data)
        
    with open(file_name, 'wb') as f: 
        f.write(data)

def read_data(file_name):
    with open(file_name, "rb") as f:
        data = f.read()
    
    #base64 to bytes
    return base64.b64decode(data)

# We are going to generate secret - public key pair in
# this stage. This will be done in the client side.

context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree = 8192,
            coeff_mod_bit_sizes = [60, 40, 40, 60]
          )

context.generate_galois_keys()
context.global_scale = 2**40

secret_context = context.serialize(save_secret_key = True)
write_data("secret.txt", secret_context)

context.make_context_public() #drop the secret_key from the context
public_context = context.serialize()
write_data("public.txt", public_context)

del context, secret_context, public_context

# We are going to apply
# homomorphic encryption to facial embeddings.
# This will be done in the client side.

# Then, homomorphic encrypted facial embeddings
# will be stored in the cloud.

context = ts.context_from(read_data("secret.txt"))

enc_v1 = ts.ckks_vector(context, img1_embedding)
enc_v2 = ts.ckks_vector(context, img2_embedding)

enc_v1_proto = enc_v1.serialize()
enc_v2_proto = enc_v2.serialize()

write_data("enc_v1.txt", enc_v1_proto)
write_data("enc_v2.txt", enc_v2_proto)

del context, enc_v1, enc_v2, enc_v1_proto, enc_v2_proto

# Once homomorphic encrypted facial embeddings 
# stored in the cloud, we are able to make 
# calculations on encrypted data.

# Notice that we just have public key here 
# and we don't have secret key. 

context = ts.context_from(read_data("public.txt"))

enc_v1_proto = read_data("enc_v1.txt")
enc_v2_proto = read_data("enc_v2.txt")

enc_v1 = ts.lazy_ckks_vector_from(enc_v1_proto)
enc_v1.link_context(context)

enc_v2 = ts.lazy_ckks_vector_from(enc_v2_proto)
enc_v2.link_context(context)

euclidean_squared = enc_v1 - enc_v2
euclidean_squared = euclidean_squared.dot(euclidean_squared)

write_data("euclidean_squared.txt", euclidean_squared.serialize())

#we must not decrypt the homomorphic encrypted euclidean squared value in this stage
#because we don't have the secret key. check this operation. it should throw an exception!

try:
    euclidean_squared.decrypt()
except Exception as err:
    print("Exception: ", str(err))

del context, enc_v1_proto, enc_v2_proto, enc_v1, enc_v2, euclidean_squared

# Once homomorphic encrypted euclidean squared value 
# found in the cloud, we are going to retrieve it to 
# the client side.

# Client can decrypt it because we have 
# the secret key in the client side

context = ts.context_from(read_data("secret.txt"))

euclidean_squared_proto = read_data("euclidean_squared.txt")

euclidean_squared = ts.lazy_ckks_vector_from(euclidean_squared_proto)
euclidean_squared.link_context(context)

euclidean_squared_plain = euclidean_squared.decrypt()[0]

print(euclidean_squared_plain)

if euclidean_squared_plain < 100:
    print("they are same person")
else:
    print("they are different persons")