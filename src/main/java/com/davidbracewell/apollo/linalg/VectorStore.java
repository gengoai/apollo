package com.davidbracewell.apollo.linalg;

import com.davidbracewell.apollo.linalg.LabeledVector;
import com.davidbracewell.apollo.linalg.ScoredLabelVector;
import com.davidbracewell.apollo.linalg.Vector;
import lombok.NonNull;

import java.util.List;
import java.util.Set;

/**
 * The interface Vector store.
 *
 * @param <KEY> the type parameter
 * @author David B. Bracewell
 */
public interface VectorStore<KEY> extends Iterable<LabeledVector> {

  /**
   * Size int.
   *
   * @return the int
   */
  int size();

  /**
   * Gets dimension.
   *
   * @return the dimension
   */
  int dimension();

  /**
   * Add.
   *
   * @param vector the vector
   */
  void add(@NonNull LabeledVector vector);

  /**
   * Query list.
   *
   * @param vector the vector
   * @return the list
   */
  List<LabeledVector> query(Vector vector);


  /**
   * Nearest list.
   *
   * @param vector    the vector
   * @param threshold the threshold
   * @return the list
   */
  List<ScoredLabelVector> nearest(Vector vector, double threshold);


  /**
   * Nearest list.
   *
   * @param vector    the vector
   * @param K         the k
   * @param threshold the threshold
   * @return the list
   */
  List<ScoredLabelVector> nearest(@NonNull Vector vector, int K, double threshold);

  /**
   * Nearest list.
   *
   * @param vector the vector
   * @return the list
   */
  List<ScoredLabelVector> nearest(Vector vector);


  /**
   * Nearest list.
   *
   * @param vector the vector
   * @param K      the k
   * @return the list
   */
  List<ScoredLabelVector> nearest(@NonNull Vector vector, int K);


  /**
   * Key set set.
   *
   * @return the set
   */
  Set<KEY> keySet();

  /**
   * Get labeled vector.
   *
   * @param key the key
   * @return the labeled vector
   */
  LabeledVector get(KEY key);

  /**
   * Contains key boolean.
   *
   * @param key the key
   * @return the boolean
   */
  boolean containsKey(KEY key);

  /**
   * Remove boolean.
   *
   * @param vector the vector
   * @return the boolean
   */
  boolean remove(LabeledVector vector);


}// END OF VectorStore
