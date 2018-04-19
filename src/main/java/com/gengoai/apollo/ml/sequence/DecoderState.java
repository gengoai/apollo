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
 */

package com.gengoai.apollo.ml.sequence;

import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.Instance;

import java.io.Serializable;
import java.util.Optional;

/**
 * <p>Structure used for recording the current decoder state.</p>
 *
 * @author David B. Bracewell
 */
public class DecoderState implements Comparable<DecoderState>, Serializable {
   private static final long serialVersionUID = 1L;
   /**
    * The Sequence probability.
    */
   public final double sequenceProbability;
   /**
    * The State probability.
    */
   public final double stateProbability;
   /**
    * The Tag.
    */
   public final String tag;
   /**
    * The Prev.
    */
   public final DecoderState previousState;
   /**
    * The Index.
    */
   public final int index;

   /**
    * Instantiates a new Decoder state.
    *
    * @param stateProbability the state probability
    * @param tag              the tag
    */
   public DecoderState(double stateProbability, String tag) {
      this(null, stateProbability, tag);
   }

   /**
    * Instantiates a new Decoder state.
    *
    * @param previousState    the previous state
    * @param stateProbability the state probability
    * @param tag              the tag
    */
   public DecoderState(DecoderState previousState, double stateProbability, String tag) {
      this.index = previousState == null ? 0 : previousState.index + 1;
      this.stateProbability = stateProbability;
      if (previousState == null) {
         this.sequenceProbability = stateProbability;//Math.log(stateProbability);
      } else {
         this.sequenceProbability = previousState.sequenceProbability + stateProbability;//Math.log(stateProbability);
      }
      this.tag = tag;
      this.previousState = previousState;
   }

   @Override
   public int compareTo(DecoderState o) {
      return Double.compare(this.sequenceProbability, o.sequenceProbability);
   }

   private class DecoderStateContext extends Context<Instance> {
      private final Sequence sequence;

      private DecoderStateContext(Sequence sequence, int index) {
         this.sequence = sequence;
         setIndex(index);
      }

      @Override
      protected Optional<Instance> getContextAt(int index) {
         return index >= 0 && index < sequence.size() ? Optional.of(sequence.get(index)) : Optional.empty();
      }

      @Override
      protected Optional<String> getLabelAt(int index) {
         if (index != DecoderState.this.index) {
            return Optional.empty();
         }
         int back = DecoderState.this.index - index;
         DecoderState ds = DecoderState.this;
         while (back > 0) {
            ds = ds.previousState;
            back--;
         }
         return Optional.ofNullable(ds).map(d -> d.tag);
      }

      @Override
      public int size() {
         return index + 1;
      }
   }

   public Context<Instance> iterator(Sequence sequence) {
      return new DecoderStateContext(sequence, index);
   }

}// END OF DecoderState
