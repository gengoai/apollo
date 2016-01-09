package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.FeatureVector;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.Optional;

/**
 * The type Feature vector sequence.
 *
 * @author David B. Bracewell
 */
public class FeatureVectorSequence implements Serializable, Iterable<FeatureVector> {
  private static final long serialVersionUID = 1L;
  private final LinkedList<FeatureVector> vectors = new LinkedList<>();

  /**
   * Add.
   *
   * @param vector the vector
   */
  public void add(FeatureVector vector) {
    if (vector != null) {
      vectors.add(vector);
    }
  }

  /**
   * Get feature vector.
   *
   * @param index the index
   * @return the feature vector
   */
  public FeatureVector get(int index) {
    return vectors.get(index);
  }

  /**
   * Size int.
   *
   * @return the int
   */
  public int size() {
    return vectors.size();
  }

  @Override
  public ContextualIterator<FeatureVector> iterator() {
    return new Iterator();
  }

  private class Iterator extends ContextualIterator<FeatureVector> {
    private static final long serialVersionUID = 1L;

    @Override
    public int size() {
      return FeatureVectorSequence.this.size();
    }

    @Override
    protected Optional<FeatureVector> getContextAt(int index) {
      if (index < size() && index >= 0) {
        return Optional.of(get(index));
      }
      return Optional.empty();
    }

    @Override
    protected Optional<String> getLabelAt(int index) {
      if (index < size() && index >= 0) {
        return Optional.of(get(index).getLabel()).map(Object::toString);
      }
      return Optional.empty();
    }
  }


}// END OF FeatureVectorSequence
