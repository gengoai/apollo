package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Instance;

import java.util.Collections;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 * The type Sequence iterator.
 *
 * @author David B. Bracewell
 */
public class SequenceIterator implements Iterator<Instance> {
  private final Sequence sequence;
  private int index = 0;
  private int current = 0;

  /**
   * Instantiates a new Sequence iterator.
   *
   * @param sequence the sequence
   */
  protected SequenceIterator(Sequence sequence) {
    this.sequence = sequence;
  }

  @Override
  public boolean hasNext() {
    return index < sequence.size();
  }

  @Override
  public Instance next() {
    if (index >= sequence.size()) {
      throw new NoSuchElementException();
    }
    current = index++;
    return sequence.get(current);
  }

  /**
   * Gets previous.
   *
   * @param n the n
   * @return the previous
   */
  public Instance getPrevious(int n) {
    return getContext(-Math.abs(n));
  }

  /**
   * Gets next.
   *
   * @param n the n
   * @return the next
   */
  public Instance getNext(int n) {
    return getContext(Math.abs(n));
  }

  /**
   * Gets context.
   *
   * @param context the context
   * @return the context
   */
  protected Instance getContext(int context) {
    int realIndex = current + context;
    if (realIndex < 0) {
      return Instance.create(Collections.emptyList(), "****START[" + realIndex + "]****");
    } else if (realIndex > sequence.size()) {
      return Instance.create(Collections.emptyList(), "****END[" + realIndex + "]****");
    }
    return sequence.get(realIndex);
  }

}// END OF SequenceIterator
