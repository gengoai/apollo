package com.davidbracewell.apollo.ml.sequence;

import java.io.Serializable;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Optional;

/**
 * The type Ctxt iterator.
 *
 * @param <T> the type parameter
 * @author David B. Bracewell
 */
public abstract class ContextualIterator<T> implements Iterator<T>, Serializable {
  private static final long serialVersionUID = 1L;
  private int index = -1;

  /**
   * Size int.
   *
   * @return the int
   */
  protected abstract int size();

  /**
   * Gets context at.
   *
   * @param index the index
   * @return the context at
   */
  protected abstract Optional<T> getContextAt(int index);

  /**
   * Gets label at.
   *
   * @param index the index
   * @return the label at
   */
  protected abstract Optional<String> getLabelAt(int index);


  /**
   * Index int.
   *
   * @return the int
   */
  public int getIndex() {
    return index;
  }


  @Override
  public boolean hasNext() {
    return (index + 1) < size();
  }

  /**
   * Next boolean.
   *
   * @return the boolean
   */
  @Override
  public T next() {
    index++;
    return getContext(0).orElseThrow(NoSuchElementException::new);
  }

  /**
   * Gets current.
   *
   * @return the current
   */
  public T getCurrent() {
    return getContextAt(index).orElseThrow(NoSuchElementException::new);
  }

  public boolean hasLabel(){
    return getLabelAt(index).isPresent();
  }

  /**
   * Gets label.
   *
   * @return the label
   */
  public String getLabel() {
    return getLabelAt(index).orElse(null);
  }

  /**
   * Gets context.
   *
   * @param relative the relative
   * @return the context
   */
  public Optional<T> getContext(int relative) {
    return getContextAt(index + relative);
  }

  /**
   * Gets context label.
   *
   * @param relative the relative
   * @return the context label
   */
  public Optional<String> getContextLabel(int relative) {
    return getLabelAt(index + relative);
  }

  /**
   * Gets previous label.
   *
   * @param relative the relative
   * @return the previous label
   */
  public Optional<String> getPreviousLabel(int relative) {
    return getContextLabel(-Math.abs(relative));
  }

  /**
   * Gets next label.
   *
   * @param relative the relative
   * @return the next label
   */
  public Optional<String> getNextLabel(int relative) {
    return getContextLabel(Math.abs(relative));
  }

  /**
   * Gets previous.
   *
   * @param relative the relative
   * @return the previous
   */
  public Optional<T> getPrevious(int relative) {
    return getContext(-Math.abs(relative));
  }

  /**
   * Gets next.
   *
   * @param relative the relative
   * @return the next
   */
  public Optional<T> getNext(int relative) {
    return getContext(Math.abs(relative));
  }

  /**
   * Has context boolean.
   *
   * @param relative the relative
   * @return the boolean
   */
  public boolean hasContext(int relative) {
    return getLabelAt(index + relative).isPresent();
  }


}// END OF ContextualIterator
