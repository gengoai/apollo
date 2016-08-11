package com.davidbracewell.apollo.ml.embedding;

import com.davidbracewell.apollo.affinity.Similarity;
import com.davidbracewell.apollo.linalg.DenseVector;
import com.davidbracewell.apollo.linalg.ScoredLabelVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.linalg.VectorStore;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Model;
import com.davidbracewell.tuple.Tuple;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static com.davidbracewell.tuple.Tuples.$;

/**
 * The type Embedding.
 *
 * @author David B. Bracewell
 */
public class Embedding implements Model, Serializable {

  private final EncoderPair encoderPair;
  private final VectorStore<String> vectorStore;

  /**
   * Instantiates a new Embedding.
   *
   * @param encoderPair the encoder pair
   * @param vectorStore the vector store
   */
  public Embedding(EncoderPair encoderPair, VectorStore<String> vectorStore) {
    this.encoderPair = encoderPair;
    this.vectorStore = vectorStore;
  }

  @Override
  public EncoderPair getEncoderPair() {
    return encoderPair;
  }


  /**
   * Gets dimension.
   *
   * @return the dimension
   */
  public int getDimension() {
    return vectorStore.dimension();
  }

  /**
   * Gets vocab.
   *
   * @return the vocab
   */
  public Set<String> getVocab() {
    return vectorStore.keySet();
  }

  /**
   * Contains boolean.
   *
   * @param word the word
   * @return the boolean
   */
  public boolean contains(String word) {
    return vectorStore.containsKey(word);
  }

  /**
   * Gets vector.
   *
   * @param word the word
   * @return the vector
   */
  public Vector getVector(String word) {
    if (contains(word)) {
      return vectorStore.get(word);
    }
    return new DenseVector(getDimension());
  }

  /**
   * Similarity double.
   *
   * @param word1 the word 1
   * @param word2 the word 2
   * @return the double
   */
  public double similarity(@NonNull String word1, @NonNull String word2) {
    Vector v1 = vectorStore.get(word1);
    Vector v2 = vectorStore.get(word2);
    if (v1 == null || v2 == null) {
      return Double.NEGATIVE_INFINITY;
    }
    return Similarity.Cosine.calculate(v1, v2);
  }

  /**
   * Nearest list.
   *
   * @param word the word
   * @param K    the k
   * @return the list
   */
  public List<ScoredLabelVector> nearest(@NonNull String word, int K) {
    return nearest(word, K, Double.NEGATIVE_INFINITY);
  }

  /**
   * Nearest list.
   *
   * @param word      the word
   * @param K         the k
   * @param threshold the threshold
   * @return the list
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

  public List<ScoredLabelVector> nearest(@NonNull Vector v, int K) {
    return nearest(v, K, Double.NEGATIVE_INFINITY);
  }


  public List<ScoredLabelVector> nearest(@NonNull Vector v, int K, double threshold) {
    if (v == null) {
      return Collections.emptyList();
    }
    return vectorStore.nearest(v, K, threshold);
  }


  /**
   * Nearest list.
   *
   * @param words the words
   * @param K     the k
   * @return the list
   */
  public List<ScoredLabelVector> nearest(@NonNull Tuple words, int K) {
    return nearest(words, $(), K, Double.NEGATIVE_INFINITY);
  }

  /**
   * Nearest list.
   *
   * @param positive  the positive
   * @param negative  the negative
   * @param K         the k
   * @param threshold the threshold
   * @return the list
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
      .nearest(pVec.subtract(nVec), K + positive.degree() + negative.degree(), threshold)
      .stream()
      .filter(slv -> !ignore.contains(slv.<String>getLabel()))
      .collect(Collectors.toList());
    return vectors.subList(0, Math.min(K, vectors.size()));
  }

}// END OF Embedding
