package com.davidbracewell.apollo.ml.sequence;

import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.function.IntFunction;

/**
 * @author David B. Bracewell
 */
public class ContextualIterator<T> implements Iterator<T> {
  private final List<T> list;
  private final IntFunction<T> startOfGenerator;
  private final IntFunction<T> endOfGenerator;
  private int index = -1;

  public ContextualIterator(List<T> list, IntFunction<T> startOfGenerator, IntFunction<T> endOfGenerator) {
    this.list = list;
    this.startOfGenerator = startOfGenerator;
    this.endOfGenerator = endOfGenerator;
  }

  public ContextualIterator(int index, List<T> list, IntFunction<T> startOfGenerator, IntFunction<T> endOfGenerator) {
    this.list = list;
    this.index = index - 1;
    this.startOfGenerator = startOfGenerator;
    this.endOfGenerator = endOfGenerator;
  }


  @Override
  public boolean hasNext() {
    return (index + 1) < list.size();
  }

  @Override
  public T next() {
    if ((index + 1) >= list.size()) {
      throw new NoSuchElementException();
    }
    index++;
    return list.get(index);
  }

  public boolean hasPrevious(int n) {
    return index - Math.abs(n) >= 0;
  }

  public boolean hasNext(int n) {
    return index + Math.abs(n) < list.size();
  }

  /**
   * Gets previous.
   *
   * @param n the n
   * @return the previous
   */
  public T getPrevious(int n) {
    return getContext(-Math.abs(n));
  }

  /**
   * Gets next.
   *
   * @param n the n
   * @return the next
   */
  public T getNext(int n) {
    return getContext(Math.abs(n));
  }

  /**
   * Gets context.
   *
   * @param context the context
   * @return the context
   */
  protected T getContext(int context) {
    int realIndex = index + context;
    if (realIndex < 0) {
      return startOfGenerator.apply(-context);
    } else if (realIndex > list.size()) {
      return endOfGenerator.apply(realIndex - list.size());
    }
    return list.get(realIndex);
  }

  public T getCurrent() {
    return list.get(index);
  }

  public int getIndex() {
    return index;
  }

}// END OF ContextualIterator
