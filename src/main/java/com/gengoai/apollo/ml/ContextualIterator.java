package com.gengoai.apollo.ml;

import java.io.Serializable;
import java.util.Iterator;

/**
 * An iterator for Examples that will iterate over both multi and single example instances facilitating easy access to
 * contextual examples.
 *
 * @author David B. Bracewell
 */
public class ContextualIterator implements Iterator<Example>, Serializable {
   private final Example example;
   private int index = -1;

   /**
    * Instantiates a new Contextual iterator to iterator over the child examples in the given Example.
    *
    * @param example the example whose children we will iterate over
    */
   public ContextualIterator(Example example) {
      this.example = example;
   }

   @Override
   public boolean hasNext() {
      return index + 1 < example.size();
   }

   @Override
   public Example next() {
      index++;
      return example.getExample(index);
   }

   /**
    * Gets the index of the current child Example
    *
    * @return the current index
    */
   public int getCurrentIndex() {
      return index;
   }


   /**
    * Gets the example a given relative position before or after the current example. When going beyond the boundaries
    * of the parent example (i.e. the relative position would be an actual <code>index < 0</code> or <code>index >=
    * parent.size()</code> a special begin of sequence or end of sequence example is returned.
    *
    * @param relativePosition the relative position (e.g. -2, -1, 1, 2)
    * @return the contextual Example at the given relative position.
    */
   public Example getContext(int relativePosition) {
      int absIndex = index + relativePosition;
      if (absIndex < 0) {
         return Instance.BEGIN_OF_SEQUENCE(absIndex);
      } else if (absIndex >= example.size()) {
         return Instance.END_OF_SEQUENCE(absIndex - example.size());
      }
      return example.getExample(absIndex);
   }


}//END OF ContextualIterator
