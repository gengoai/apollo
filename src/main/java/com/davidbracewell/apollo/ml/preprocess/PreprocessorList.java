package com.davidbracewell.apollo.ml.preprocess;

import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Example;
import com.davidbracewell.conversion.Cast;
import com.google.common.collect.Iterators;
import lombok.NonNull;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

/**
 * The type Preprocessor list.
 *
 * @param <T> the type parameter
 */
public class PreprocessorList<T extends Example> implements Iterable<Preprocessor<T>>, Serializable {
  private static final long serialVersionUID = 1L;
  private final List<Preprocessor<T>> preprocessors = new LinkedList<>();
  private boolean finished = false;


  /**
   * Instantiates a new Preprocessor list.
   *
   * @param preprocessors the preprocessors
   */
  public PreprocessorList(@NonNull Collection<? extends Preprocessor<T>> preprocessors) {
    this.preprocessors.addAll(preprocessors);
  }

  public static <T extends Example> PreprocessorList<T> empty() {
    return new PreprocessorList<>(Collections.emptyList());
  }

  /**
   * Create preprocessor list.
   *
   * @param <T>           the type parameter
   * @param preprocessors the preprocessors
   * @return the preprocessor list
   */
  @SafeVarargs
  public static <T extends Example> PreprocessorList<T> create(Preprocessor<T>... preprocessors) {
    PreprocessorList<T> list;
    if (preprocessors == null) {
      list = new PreprocessorList<>(Collections.emptyList());
    } else {
      list = new PreprocessorList<>(Arrays.asList(preprocessors));
    }
    return list;
  }

  @Override
  public Iterator<Preprocessor<T>> iterator() {
    return Iterators.unmodifiableIterator(preprocessors.iterator());
  }

  /**
   * Visit.
   *
   * @param examplesIterator the examples iterator
   */
  public void visit(Iterator<T> examplesIterator) {
    if (examplesIterator != null) {
      examplesIterator.forEachRemaining(this::visit);
    }
  }

  /**
   * Visit.
   *
   * @param example the example
   */
  public void visit(T example) {
    if (!finished) {
      if (example != null) {
        preprocessors.forEach(p -> p.visit(example));
      }
    }
  }

  public boolean isEmpty() {
    return preprocessors.isEmpty();
  }

  public int size() {
    return preprocessors.size();
  }

  /**
   * Finish.
   */
  public void finish() {
    if (!finished) {
      preprocessors.forEach(Preprocessor::finish);
      finished = true;
    }
  }

  /**
   * Is finished boolean.
   *
   * @return the boolean
   */
  public boolean isFinished() {
    return finished;
  }

  /**
   * Apply t.
   *
   * @param example the example
   * @return the t
   */
  public T apply(T example) {
    if (isEmpty()) {
      return example;
    }
    T transformed = Cast.as(example.copy());
    for (Preprocessor<T> preprocessor : preprocessors) {
      transformed = preprocessor.process(transformed);
    }
    return transformed;
  }

  /**
   * Gets runtime only.
   *
   * @return the runtime only
   */
  public PreprocessorList<T> getModelProcessors() {
    return new PreprocessorList<>(
      preprocessors.stream()
        .filter(p -> !p.trainOnly())
        .collect(Collectors.toList()));
  }

  public void trimToSize(@NonNull Encoder encoder) {
    preprocessors.forEach(p -> p.trimToSize(encoder));
  }

  public void reset() {
    this.finished = false;
    preprocessors.forEach(Preprocessor::reset);
  }

}// END OF PreprocessorList
