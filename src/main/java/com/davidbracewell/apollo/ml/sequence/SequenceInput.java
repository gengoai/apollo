package com.davidbracewell.apollo.ml.sequence;

import com.google.common.collect.ForwardingList;
import lombok.NonNull;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.function.IntFunction;

/**
 * @author David B. Bracewell
 */
public class SequenceInput<T> extends ForwardingList<T> implements Serializable {
  private static final long serialVersionUID = 1L;
  private final ArrayList<T> list;

  public SequenceInput(@NonNull Collection<T> list) {
    this.list = new ArrayList<>(list);
    this.list.trimToSize();
  }

  @Override
  protected List<T> delegate() {
    return list;
  }


  public ContextualIterator<T> iterator(IntFunction<T> startOfGenerator, IntFunction<T> endOfGenerator) {
    return new ContextualIterator<>(
      list,
      startOfGenerator,
      endOfGenerator
    );
  }


}// END OF SequenceInput
