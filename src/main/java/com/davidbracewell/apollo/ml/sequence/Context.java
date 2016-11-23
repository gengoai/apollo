package com.davidbracewell.apollo.ml.sequence;

import java.io.Serializable;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.Optional;

/**
 * <p>A specialized iterator for working with sequences that allows access to the context of the current element.</p>
 *
 * @param <T> the input type parameter
 * @author David B. Bracewell
 */
public abstract class Context<T> implements Iterator<T>, Serializable {
   private static final long serialVersionUID = 1L;
   private int index = -1;

   /**
    * The number of items in the iterator
    *
    * @return the number of items in the iterator
    */
   public abstract int size();

   /**
    * Gets the element at the given index (absolute)
    *
    * @param index the index
    * @return the context
    */
   protected abstract Optional<T> getContextAt(int index);

   /**
    * Gets the label for the element at the given index (absolute)
    *
    * @param index the index
    * @return the label
    */
   protected abstract Optional<String> getLabelAt(int index);


   /**
    * Gets the index of the current item in the iterator
    *
    * @return the index of the current item
    */
   public int getIndex() {
      return index;
   }


   @Override
   public boolean hasNext() {
      return (index + 1) < size();
   }

   /**
    * Gets the next item in the context
    *
    * @return the next item in the context
    */
   @Override
   public T next() {
      index++;
      return getContext(0).orElseThrow(NoSuchElementException::new);
   }

   /**
    * Gets the current item.
    *
    * @return the current item
    */
   public T getCurrent() {
      return getContextAt(index).orElseThrow(NoSuchElementException::new);
   }

   /**
    * Gets the label of the current item.
    *
    * @return the label or null if none
    */
   public String getLabel() {
      return getLabelAt(index).orElse(null);
   }

   /**
    * Gets the context relative to the current item.
    *
    * @param relative the relative index (negative numbers mean previous, positive mean next)
    * @return the context
    */
   public Optional<T> getContext(int relative) {
      return getContextAt(index + relative);
   }

   /**
    * Gets the label relative to the current item.
    *
    * @param relative the relative index (negative numbers mean previous, positive mean next)
    * @return the label
    */
   public Optional<String> getContextLabel(int relative) {
      return getLabelAt(index + relative);
   }

   /**
    * Gets the <code>relative</code> previous label.
    *
    * @param relative the relative index
    * @return the previous label
    */
   public Optional<String> getPreviousLabel(int relative) {
      return getContextLabel(-Math.abs(relative));
   }

   /**
    * Gets the <code>relative</code> next label.
    *
    * @param relative the relative index
    * @return the next label
    */
   public Optional<String> getNextLabel(int relative) {
      return getContextLabel(Math.abs(relative));
   }

   /**
    * Gets the <code>relative</code> previous element.
    *
    * @param relative the relative index
    * @return the previous element
    */
   public Optional<T> getPrevious(int relative) {
      return getContext(-Math.abs(relative));
   }

   /**
    * Gets the <code>relative</code> next element.
    *
    * @param relative the relative index
    * @return the next element
    */
   public Optional<T> getNext(int relative) {
      return getContext(Math.abs(relative));
   }

}// END OF Context
