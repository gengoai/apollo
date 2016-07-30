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
 * @author David B. Bracewell
 */
public class Embedding implements Model, Serializable {

  private final EncoderPair encoderPair;
  private final VectorStore<String> vectorStore;

  public Embedding(EncoderPair encoderPair, VectorStore<String> vectorStore) {
    this.encoderPair = encoderPair;
    this.vectorStore = vectorStore;
  }

  @Override
  public EncoderPair getEncoderPair() {
    return encoderPair;
  }


  public int getDimension() {
    return vectorStore.dimension();
  }

  public Set<String> getVocab() {
    return vectorStore.keySet();
  }

  public boolean contains(String word) {
    return vectorStore.containsKey(word);
  }

  public Vector getVector(String word) {
    if (contains(word)) {
      return vectorStore.get(word);
    }
    return new DenseVector(getDimension());
  }

  public double similarity(@NonNull String word1, @NonNull String word2) {
    Vector v1 = vectorStore.get(word1);
    Vector v2 = vectorStore.get(word2);
    if (v1 == null || v2 == null) {
      return Double.NEGATIVE_INFINITY;
    }
    return Similarity.Cosine.calculate(v1, v2);
  }

  public List<ScoredLabelVector> nearest(@NonNull String word, int K) {
    return nearest(word, K, Double.NEGATIVE_INFINITY);
  }

  public List<ScoredLabelVector> nearest(@NonNull String word, int K, double threshold) {
    Vector v1 = vectorStore.get(word);
    if (v1 == null) {
      return Collections.emptyList();
    }
    return vectorStore.nearest(v1, K + 1, threshold).stream().filter(slv -> !word.equals(slv.getLabel())).collect(Collectors.toList());
  }

  public List<ScoredLabelVector> nearest(@NonNull Tuple words, int K) {
    return nearest(words, $(), K, Double.NEGATIVE_INFINITY);
  }

  public List<ScoredLabelVector> nearest(@NonNull Tuple positive, @NonNull Tuple negative, int K, double threshold) {
    Vector pVec = new DenseVector(getDimension());
    positive.forEach(word -> pVec.addSelf(getVector(word.toString())));
    Vector nVec = new DenseVector(getDimension());
    negative.forEach(word -> nVec.addSelf(getVector(word.toString())));
    Set<String> ignore = new HashSet<>();
    positive.forEach(o -> ignore.add(o.toString()));
    negative.forEach(o -> ignore.add(o.toString()));
    List<ScoredLabelVector> vectors = vectorStore.nearest(pVec.subtract(nVec), K + positive.degree() + negative.degree(), threshold).stream().filter(slv -> !ignore.contains(slv.<String>getLabel())).collect(Collectors.toList());
    return vectors.subList(0, Math.min(K,vectors.size()));
  }

}// END OF Embedding
