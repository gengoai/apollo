/*
 * (c) 2005 David B. Bracewell
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 */

package com.gengoai.apollo.ml;

import java.io.Serializable;
import java.util.Iterator;

/**
 * <p>
 * An Iterator over examples allowing easy access to the current {@link Example}s context using a relative offset. When
 * requesting an {@link Example} at an offset index out of bounds a dummy beginning of sequence or end of sequence
 * example is returned.
 * </p>
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

   /**
    * Gets the index of the current child Example
    *
    * @return the current index
    */
   public int getCurrentIndex() {
      return index;
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


}//END OF ContextualIterator
