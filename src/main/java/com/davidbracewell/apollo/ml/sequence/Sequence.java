package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Example;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.collection.Interner;
import lombok.NonNull;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * The type Sequence.
 *
 * @author David B. Bracewell
 */
public class Sequence implements Example, Serializable {
  private static final long serialVersionUID = 1L;
  private final List<Instance> sequence;

  /**
   * Instantiates a new Sequence.
   *
   * @param sequence the sequence
   */
  public Sequence(@NonNull List<Instance> sequence) {
    this.sequence = sequence;
  }

  @Override
  public Stream<String> getFeatureSpace() {
    return sequence.stream().flatMap(Instance::getFeatureSpace).distinct();
  }

  @Override
  public Stream<Object> getLabelSpace() {
    return sequence.stream().flatMap(Instance::getLabelSpace).distinct();
  }

  @Override
  public Sequence intern(Interner<String> interner) {
    return new Sequence(
      sequence.stream().map(instance -> instance.intern(interner)).collect(Collectors.toList())
    );
  }

  @Override
  public Sequence copy() {
    return new Sequence(
      sequence.stream().map(Instance::copy).collect(Collectors.toList())
    );
  }

  /**
   * Get instance.
   *
   * @param index the index
   * @return the instance
   */
  public Instance get(int index) {
    return sequence.get(index);
  }

  /**
   * Size int.
   *
   * @return the int
   */
  public int size() {
    return sequence.size();
  }

  /**
   * Iterator sequence iterator.
   *
   * @return the sequence iterator
   */
  public ContextualIterator<Instance> iterator() {
    return new ContextualIterator<>(
      asInstances(),
      i -> Instance.create(Collections.emptyList(), "****START****"),
      i -> Instance.create(Collections.emptyList(), "****END****")
    );
  }

  @Override
  public List<Instance> asInstances() {
    return sequence;
  }

  @Override
  public String toString() {
    return sequence.toString();
  }

}// END OF Sequence
