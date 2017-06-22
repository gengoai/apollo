package com.davidbracewell.apollo.ml.embedding;

import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.linalg.VectorComposition;
import com.davidbracewell.apollo.linalg.store.CosineSignature;
import com.davidbracewell.apollo.linalg.store.InMemoryLSH;
import com.davidbracewell.apollo.linalg.store.VectorStore;
import com.davidbracewell.apollo.ml.*;
import com.davidbracewell.conversion.Cast;
import com.davidbracewell.function.SerializableSupplier;
import com.davidbracewell.function.Unchecked;
import com.davidbracewell.guava.common.primitives.Ints;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.tuple.Tuple;
import lombok.NonNull;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

import static com.davidbracewell.tuple.Tuples.$;

/**
 * <p>A word/feature embedding model where words/features have been mapped into vectors.</p>
 *
 * @author David B. Bracewell
 */
public class Embedding implements Model, Serializable {

   private static final long serialVersionUID = -5216687988129520383L;
   private final EncoderPair encoderPair;
   private VectorStore<String> vectorStore;

   /**
    * Instantiates a new Embedding.
    *
    * @param encoderPair the encoder pair
    * @param vectorStore the vector store
    */
   public Embedding(@NonNull EncoderPair encoderPair, @NonNull VectorStore<String> vectorStore) {
      this.encoderPair = encoderPair;
      this.vectorStore = vectorStore;
   }

   /**
    * From vector store embedding.
    *
    * @param vectors the vectors
    * @return the embedding
    */
   public static Embedding fromVectorStore(@NonNull VectorStore<String> vectors) {
      return new Embedding(new EncoderPair(new LabelIndexEncoder(), new IndexEncoder()), vectors);
   }

   public static Embedding fromWord2VecTextFile(@NonNull Resource in) throws IOException {
      List<String> lines = in.readLines();
      int firstRow = 1;
      Integer dimension = Ints.tryParse(lines.get(0).split("\\s+")[1]);
      if (dimension == null) {
         firstRow = 0;
         dimension = lines.get(0).trim().split("\\s+").length - 1;
      }
      VectorStore<String> vectorStore = InMemoryLSH.builder()
                                                   .dimension(dimension)
                                                   .signatureSupplier(CosineSignature::new)
                                                   .createVectorStore();
      lines.stream()
           .skip(firstRow)
           .parallel()
           .map(line -> {
              Vector v = new DenseVector(vectorStore.dimension());
              String[] parts = line.trim().split("\\s+");
              for (int vi = 1; vi < parts.length; vi++) {
                 v.set(vi - 1, Double.parseDouble(parts[vi]));
              }
              v.setLabel(parts[0].replace('_', ' '));
              return v;
           })
           .collect(Collectors.toList())
           .forEach(vectorStore::add);
      return new Embedding(new EncoderPair(new NoOptLabelEncoder(), new NoOptEncoder()), vectorStore);
   }


   /**
    * Reads the model from the given resource
    *
    * @param modelResource the resource containing the serialized model
    * @return the deserialized model
    * @throws Exception Something went wrong reading the model
    */
   public static Embedding read(@NonNull Resource modelResource) throws Exception {
      Object o = modelResource.readObject();
      if (o instanceof Embedding) {
         return Cast.as(o);
      } else if (o instanceof VectorStore) {
         return Embedding.fromVectorStore(Cast.as(o));
      }
      throw new IllegalArgumentException(o.getClass() + " cannot be read in as an embedding");
   }

   public static SerializableSupplier<Embedding> supplier(final boolean word2vecFormat, @NonNull final Resource resource) {
      if (word2vecFormat) {
         return Unchecked.supplier(() -> Embedding.fromWord2VecTextFile(resource));
      }
      return Unchecked.supplier(() -> Embedding.read(resource));
   }

   /**
    * Creates a vector using the given vector composition for the given words.
    *
    * @param composition the composition function to use
    * @param words       the words whose vectors we want to compose
    * @return a composite vector consisting of the given words and calculated using the given vector composition
    */
   public Vector compose(@NonNull VectorComposition composition, String... words) {
      if (words == null) {
         return new SparseVector(getDimension());
      } else if (words.length == 1) {
         return getVector(words[0]);
      }
      List<Vector> vectors = new ArrayList<>();
      for (String w : words) {
         vectors.add(getVector(w));
      }
      return composition.compose(getDimension(), vectors);
   }

   /**
    * Checks if the given word/feature is present in the embedding
    *
    * @param word the word/feature to check
    * @return True if the word/feature is in the embedding, False otherwise
    */
   public boolean contains(String word) {
      return vectorStore.containsKey(word);
   }

   /**
    * Gets the dimension of the vectors in the embedding.
    *
    * @return the vector dimension
    */
   public int getDimension() {
      return vectorStore.dimension();
   }

   @Override
   public EncoderPair getEncoderPair() {
      return encoderPair;
   }

   /**
    * Gets the vector for the given word/feature.
    *
    * @param word the word/feature whose vector is to be retrieved
    * @return the vector for the given word/feature or a zero-vector if the word/feature is not in the embedding.
    */
   public Vector getVector(String word) {
      if (contains(word)) {
         return vectorStore.get(word);
      }
      return new DenseVector(getDimension());
   }

   /**
    * Gets the underlying vector store
    *
    * @return The underlying vector store
    */
   public VectorStore<String> getVectorStore() {
      return vectorStore;
   }

   /**
    * Gets the vocabulary of words/features with vectors.
    *
    * @return the vocabulary
    */
   public Set<String> getVocab() {
      return vectorStore.keySet();
   }

   /**
    * Finds the closest K vectors to the given word/feature in the embedding
    *
    * @param word the word/feature whose neighbors we want
    * @param K    the maximum number of neighbors to return
    * @return the list of scored K-nearest vectors
    */
   public List<Vector> nearest(@NonNull String word, int K) {
      return nearest(word, K, Double.NEGATIVE_INFINITY);
   }

   /**
    * Finds the closest K vectors to the given word/feature in the embedding
    *
    * @param word      the word/feature whose neighbors we want
    * @param K         the maximum number of neighbors to return
    * @param threshold threshold for selecting vectors
    * @return the list of scored K-nearest vectors
    */
   public List<Vector> nearest(@NonNull String word, int K, double threshold) {
      Vector v1 = vectorStore.get(word);
      if (v1 == null) {
         return Collections.emptyList();
      }
      List<Vector> near = vectorStore
                             .nearest(v1, K + 1, threshold)
                             .stream()
                             .filter(slv -> !word.equals(slv.getLabel()))
                             .collect(Collectors.toList());
      return near.subList(0, Math.min(K, near.size()));
   }

   /**
    * Finds the closest K vectors to the given vector in the embedding
    *
    * @param v the vector whose neighbors we want
    * @param K the maximum number of neighbors to return
    * @return the list of scored K-nearest vectors
    */
   public List<Vector> nearest(@NonNull Vector v, int K) {
      return nearest(v, K, Double.NEGATIVE_INFINITY);
   }

   /**
    * Finds the closest K vectors to the vector in the embedding
    *
    * @param v         the vector whose neighbors we want
    * @param K         the maximum number of neighbors to return
    * @param threshold threshold for selecting vectors
    * @return the list of scored K-nearest vectors
    */
   public List<Vector> nearest(@NonNull Vector v, int K, double threshold) {
      return vectorStore.nearest(v, K, threshold);
   }

   /**
    * Finds the closest K vectors to the given tuple of words/features in the embedding
    *
    * @param words a tuple of words/features (the individual vectors are composed using vector addition) whose neighbors
    *              we want
    * @param K     the maximum number of neighbors to return
    * @return the list of scored K-nearest vectors
    */
   public List<Vector> nearest(@NonNull Tuple words, int K) {
      return nearest(words, $(), K, Double.NEGATIVE_INFINITY);
   }

   /**
    * Finds the closest K vectors to the given positive tuple of words/features and not near the negative tuple of
    * words/features in the embedding
    *
    * @param positive  a tuple of words/features (the individual vectors are composed using vector addition) whose
    *                  neighbors we want
    * @param negative  a tuple of words/features (the individual vectors are composed using vector addition) subtracted
    *                  from the positive vectors.
    * @param K         the maximum number of neighbors to return
    * @param threshold threshold for selecting vectors
    * @return the list of scored K-nearest vectors
    */
   public List<Vector> nearest(@NonNull Tuple positive, @NonNull Tuple negative, int K, double threshold) {
      Vector pVec = new DenseVector(getDimension());
      positive.forEach(word -> pVec.addSelf(getVector(word.toString())));
      Vector nVec = new DenseVector(getDimension());
      negative.forEach(word -> nVec.addSelf(getVector(word.toString())));
      Set<String> ignore = new HashSet<>();
      positive.forEach(o -> ignore.add(o.toString()));
      negative.forEach(o -> ignore.add(o.toString()));
      List<Vector> vectors = vectorStore
                                .nearest(pVec.subtract(nVec), K + positive.degree() + negative.degree(),
                                         threshold)
                                .stream()
                                .filter(slv -> !ignore.contains(slv.<String>getLabel()))
                                .collect(Collectors.toList());
      return vectors.subList(0, Math.min(K, vectors.size()));
   }

}// END OF Embedding
