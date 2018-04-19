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

/**
 * <p>Greedy decoder that optimizes locally instead of globally.</p>
 *
 * @author David B. Bracewell
 */
public class WindowDecoder implements Decoder, Serializable {
   private static final long serialVersionUID = 1L;

   @Override
   public Labeling decode(SequenceLabeler labeler, Sequence sequence) {
      Labeling result = new Labeling(sequence.size());
      DecoderState state = null;
      String previousLabel = null;

      for (Context<Instance> iterator = sequence.iterator(); iterator.hasNext(); ) {
         if (state == null) {
            state = new DecoderState(null, 0d, null);
         }
         double[] results = labeler.estimate(
            iterator.next().getFeatures().iterator(),
            labeler.getTransitionFeatures().extract(state.iterator(sequence)));

         double max = Double.NEGATIVE_INFINITY;
         String label = null;
         for (int i = 0; i < results.length; i++) {
            String tL = labeler.getEncoderPair().decodeLabel(i).toString();
            if (results[i] > max && labeler.getValidator().isValid(tL, previousLabel, iterator.getCurrent())) {
               max = results[i];
               label = tL;
            }
         }
         if (max == Double.NEGATIVE_INFINITY) {
            for (int i = 0; i < results.length; i++) {
               String tL = labeler.getEncoderPair().decodeLabel(i).toString();
               if (results[i] > max) {
                  max = results[i];
                  label = tL;
               }
            }
         }
         previousLabel = label;
         result.setLabel(iterator.getIndex(), label, max);
         state = new DecoderState(state, max, label);
      }
      return result;
   }

}// END OF WindowDecoder
