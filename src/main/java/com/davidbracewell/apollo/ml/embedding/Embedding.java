package com.davidbracewell.apollo.ml.embedding;

import com.davidbracewell.apollo.linalg.*;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.linalg.store.VectorStore;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.IndexEncoder;
import com.davidbracewell.apollo.ml.LabelIndexEncoder;
import com.davidbracewell.apollo.ml.Model;
import com.davidbracewell.tuple.Tuple;
import lombok.NonNull;

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

   @Override
   public EncoderPair getEncoderPair() {
      return encoderPair;
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

   /**
    * Gets the underlying vector store
    *
    * @return The underlying vector store
    */
   public VectorStore<String> getVectorStore() {
      return vectorStore;
   }

   /**
    * Gets the dimension of the vectors in the embedding.
    *
    * @return the vector dimension
    */
   public int getDimension() {
      return vectorStore.dimension();
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
    * Checks if the given word/feature is present in the embedding
    *
    * @param word the word/feature to check
    * @return True if the word/feature is in the embedding, False otherwise
    */
   public boolean contains(String word) {
      return vectorStore.containsKey(word);
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
    * Finds the closest K vectors to the given word/feature in the embedding
    *
    * @param word the word/feature whose neighbors we want
    * @param K    the maximum number of neighbors to return
    * @return the list of scored K-nearest vectors
    */
   public List<ScoredLabelVector> nearest(@NonNull String word, int K) {
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
   public List<ScoredLabelVector> nearest(@NonNull String word, int K, double threshold) {
      Vector v1 = vectorStore.get(word);
      if (v1 == null) {
         return Collections.emptyList();
      }
      return vectorStore
                .nearest(v1, K + 1, threshold)
                .stream()
                .filter(slv -> !word.equals(slv.getLabel()))
                .collect(Collectors.toList());
   }


   /**
    * Finds the closest K vectors to the given vector in the embedding
    *
    * @param v the vector whose neighbors we want
    * @param K the maximum number of neighbors to return
    * @return the list of scored K-nearest vectors
    */
   public List<ScoredLabelVector> nearest(@NonNull Vector v, int K) {
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
   public List<ScoredLabelVector> nearest(@NonNull Vector v, int K, double threshold) {
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
   public List<ScoredLabelVector> nearest(@NonNull Tuple words, int K) {
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
   public List<ScoredLabelVector> nearest(@NonNull Tuple positive, @NonNull Tuple negative, int K, double threshold) {
      Vector pVec = new DenseVector(getDimension());
      positive.forEach(word -> pVec.addSelf(getVector(word.toString())));
      Vector nVec = new DenseVector(getDimension());
      negative.forEach(word -> nVec.addSelf(getVector(word.toString())));
      Set<String> ignore = new HashSet<>();
      positive.forEach(o -> ignore.add(o.toString()));
      negative.forEach(o -> ignore.add(o.toString()));
      List<ScoredLabelVector> vectors = vectorStore
                                           .nearest(pVec.subtract(nVec), K + positive.degree() + negative.degree(),
                                                    threshold)
                                           .stream()
                                           .filter(slv -> !ignore.contains(slv.<String>getLabel()))
                                           .collect(Collectors.toList());
      return vectors.subList(0, Math.min(K, vectors.size()));
   }

}// END OF Embedding
